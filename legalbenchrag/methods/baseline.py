import os
import sqlite3
import struct
from typing import Literal, cast, List

import sqlite_vec
from pydantic import BaseModel
from tqdm import tqdm

from legalbenchrag.benchmark_types import (
    Document,
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from legalbenchrag.utils.ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    AIRerankModel,
    ai_embedding,
    ai_rerank,
)
from legalbenchrag.utils.chunking import Chunk, get_chunks


def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)


SHOW_LOADING_BAR = True


class ChunkingStrategy(BaseModel):
    # Use strategy_name defined in chunking.py
    strategy_name: Literal["naive", "rcts"]
    chunk_size: int
    # Add overlap config if rcts uses it
    chunk_overlap_ratio: float = 0.0


class RetrievalStrategy(BaseModel):
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_topk: int
    rerank_model: AIRerankModel | None
    rerank_topk: int
    token_limit: int | None


class EmbeddingInfo(BaseModel):
    """Stores metadata associated with a specific vector rowid."""
    document_id: str  # file_path from Chunk
    span: tuple[int, int]


class BaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    # This list maps sqlite rowid (implicit index) to metadata
    embedding_infos: List[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None

    async def cleanup(self) -> None:
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path is not None and os.path.exists(
            self.sqlite_db_file_path
        ):
            try:
                os.remove(self.sqlite_db_file_path)
            except OSError as e:
                print(f"Warning: Could not remove baseline DB file: {e}")
            self.sqlite_db_file_path = None

    async def ingest_document(self, document: Document) -> None:
        # Store the full document content for later text retrieval
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        # 1. Calculate chunks using the shared utility
        print("Baseline: Calculating chunks...")
        all_chunks: List[Chunk] = []
        for document in self.documents.values():
            chunks_for_doc = get_chunks(
                document=document,
                strategy_name=self.retrieval_strategy.chunking_strategy.strategy_name,
                chunk_size=self.retrieval_strategy.chunking_strategy.chunk_size,
                chunk_overlap_ratio=self.retrieval_strategy.chunking_strategy.chunk_overlap_ratio,
            )
            all_chunks.extend(chunks_for_doc)
        print(f"Baseline: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("Baseline: No chunks created, skipping embedding and indexing.")
            self.embedding_infos = []
            return

        # Prepare list for metadata mapping (index will correspond to rowid)
        self.embedding_infos = []

        # 2. Calculate embeddings using the ai_embedding function
        progress_bar: tqdm | None = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(
                total=len(all_chunks), desc="Baseline: Processing Embeddings", ncols=100
            )

        # Define callback for progress update
        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # Get embeddings for all chunk contents
        chunk_contents = [chunk.content for chunk in all_chunks]
        embeddings = await ai_embedding(
            self.retrieval_strategy.embedding_model,
            chunk_contents,
            AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )

        if progress_bar:
            progress_bar.close()

        assert len(all_chunks) == len(embeddings), "Mismatch between chunks and embeddings count"

        print(f"Baseline: Start indexing embeddings...")

        # 3. Store embeddings and metadata in SQLite-Vec
        if self.sqlite_db is None:
            # random_id = str(uuid4())  # Not needed if we always overwrite/delete
            self.sqlite_db_file_path = f"./data/cache/baseline_{os.getpid()}.db"  # Add PID to avoid multi-process conflicts if run in parallel later
            if os.path.exists(self.sqlite_db_file_path):
                os.remove(self.sqlite_db_file_path)
            self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
            self.sqlite_db.enable_load_extension(True)
            sqlite_vec.load(self.sqlite_db)
            self.sqlite_db.enable_load_extension(False)
            # Set RAM Usage and create vector table
            self.sqlite_db.execute(f"PRAGMA mmap_size = {3*1024*1024*1024}")  # 3GB RAM usage limit
            self.sqlite_db.execute(
                f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
            )

        with self.sqlite_db as db:
            insert_data = []
            current_rowid = 0
            for chunk, embedding in zip(all_chunks, embeddings):
                # Add metadata to our mapping list BEFORE inserting, index matches future rowid
                self.embedding_infos.append(
                    EmbeddingInfo(
                        document_id=chunk.file_path,
                        span=chunk.span,
                    )
                )
                # Prepare data for insertion (rowid will be index + 1) -> sqlite rowids start at 1 typically
                insert_data.append(
                    (current_rowid + 1, serialize_f32(embedding))
                )
                current_rowid += 1

            # Insert into sqlite-vec
            db.executemany(
                # Let sqlite handle rowid auto-increment
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                insert_data,
            )

        print(f"Baseline: Finished indexing {len(self.embedding_infos)} embeddings.")

    async def query(self, query: str) -> QueryResponse:
        if self.sqlite_db is None or self.embedding_infos is None:
            raise ValueError("Sync documents before querying!")

        # 1. Get TopK Embedding results
        query_embedding = (
            await ai_embedding(
                self.retrieval_strategy.embedding_model, [query], AIEmbeddingType.QUERY
            )
        )[0]

        rows = self.sqlite_db.execute(
            """
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            ORDER BY distance ASC
            LIMIT ?
            """,
            [serialize_f32(query_embedding), self.retrieval_strategy.embedding_topk],
        ).fetchall()

        # Get metadata using the rowid (adjusting for 0-based list index -> sqlite rowid starts at 1)
        retrieved_indices = [cast(int, row[0]) - 1 for row in rows]  # Adjust to 0-based index
        retrieved_metadatas = [self.embedding_infos[i] for i in retrieved_indices]
        initial_texts = [self.get_embedding_info_text(meta) for meta in retrieved_metadatas]

        # 2. Rerank if specified
        final_metadatas = retrieved_metadatas
        if self.retrieval_strategy.rerank_model is not None and initial_texts:
            reranked_indices_map = await ai_rerank(
                self.retrieval_strategy.rerank_model,
                query,
                texts=initial_texts,
                top_k=self.retrieval_strategy.rerank_topk,
            )
            # The reranked indices refer to the order in `initial_texts`
            final_metadatas = [
                retrieved_metadatas[i] for i in reranked_indices_map
            ]
            # No need to slice again by rerank_topk, ai_rerank already handles it
            # MR: was: reranked_indices[: self.retrieval_strategy.rerank_topk]

        # 3. Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets: list[RetrievedSnippet] = []
        for i, metadata in enumerate(final_metadatas):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            # Use the span directly from the metadata
            span = metadata.span
            text_content = self.get_embedding_info_text(metadata) # Get text for length check
            current_len = len(text_content)

            final_span = span
            if remaining_tokens is not None:
                if current_len > remaining_tokens:
                    # Truncate span if exceeding remaining tokens
                    final_span = (span[0], span[0] + remaining_tokens)
                    current_len = remaining_tokens  # Update length used
                remaining_tokens -= current_len  # Deduct used length

            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=metadata.document_id,
                    span=final_span,
                    score=1.0 / (i + 1),  # Simple rank-based score
                )
            )

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        """Retrieves the text content for a given EmbeddingInfo."""
        # Use stored full documents
        return self.documents[embedding_info.document_id].content[
            embedding_info.span[0] : embedding_info.span[1]
        ]