import os
import sqlite3
import struct
from typing import cast, List, Dict
import asyncio
import logging
from pathlib import Path

import sqlite_vec
from pydantic import BaseModel
from tqdm import tqdm

from sac_rag.data_models import (
    Document,
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from sac_rag.utils.ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    AIRerankModel,
    ai_embedding,
    ai_rerank,
    generate_document_summary
)
from sac_rag.utils.chunking import Chunk, get_chunks, ChunkingStrategy
from sac_rag.utils.stats_tracker import stats_tracker

logger = logging.getLogger(__name__)


def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)


SHOW_LOADING_BAR = True


class BaselineRetrievalStrategy(BaseModel):
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_top_k: int
    rerank_model: AIRerankModel | None
    rerank_top_k: List[int]
    token_limit: int | None


class EmbeddingInfo(BaseModel):
    """Stores metadata associated with a specific vector rowid."""
    document_id: str  # file_path from Chunk
    span: tuple[int, int]  # Span of the original content part in the original document
    # TODO: Remove processed_content later. Right now it serves to write the summary+chunk in the json files.
    #  This is memory heavy!
    processed_content: str  # The full text that was embedded: (summary +) original


class BaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: BaselineRetrievalStrategy
    documents: dict[str, Document]
    # This list maps sqlite rowid (implicit index) to metadata
    embedding_infos: List[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | Path | None

    def __init__(self, retrieval_strategy: BaselineRetrievalStrategy, cache_dir: str | Path | None = None):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.cache_dir = Path.cwd() / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # The filename is now constructed from the configured, absolute path
        self.sqlite_db_file_path = self.cache_dir / f"baseline_{os.getpid()}.db"

    async def cleanup(self) -> None:
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path is not None and os.path.exists(self.sqlite_db_file_path):
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
        stats_tracker.start_timer('chunking_and_summarization')
        print(f"Baseline: Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")  # Updated print

        # Prepare kwargs for the now async get_chunks
        chunking_params = {
            "strategy_name": self.retrieval_strategy.chunking_strategy.strategy_name,
            "chunk_size": self.retrieval_strategy.chunking_strategy.chunk_size,  # Total size for summary strats
            "chunk_overlap_ratio": self.retrieval_strategy.chunking_strategy.chunk_overlap_ratio,
        }
        # Add summarization specific parameters if it's a summary strategy
        if self.retrieval_strategy.chunking_strategy.strategy_name.startswith("summary_"):
            chunking_params["summarization_model"] = self.retrieval_strategy.chunking_strategy.summary_model
            chunking_params["summary_prompt_template"] = self.retrieval_strategy.chunking_strategy.summary_prompt_template
            chunking_params["prompt_target_char_length"] = self.retrieval_strategy.chunking_strategy.prompt_target_char_length
            chunking_params["summary_truncation_length"] = self.retrieval_strategy.chunking_strategy.summary_truncation_length
            chunking_params["use_cache"] = self.retrieval_strategy.chunking_strategy.use_cache

        all_chunks_tasks: List[asyncio.Task[List[Chunk]]] = []
        for document in self.documents.values():
            # get_chunks is now async
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))  # type: ignore
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)

        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"Baseline: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("Baseline: No chunks created, skipping embedding and indexing.")
            self.embedding_infos = []  # Ensure embedding_infos is initialized
            return

        # Prepare list for metadata mapping (index will correspond to rowid)
        self.embedding_infos = []  # Initialize here

        # 2. Calculate embeddings using the ai_embedding function
        stats_tracker.start_timer('embedding_generation')
        progress_bar: tqdm | None = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(
                total=len(all_chunks), desc="Baseline: Processing Embeddings", ncols=100
            )

        # Define callback for progress update
        def progress_callback():
            if progress_bar:
                progress_bar.update(1)

        # Get embeddings for all chunk contents (which now include summaries if applicable)
        chunk_contents = [chunk.content for chunk in all_chunks]
        embeddings = await ai_embedding(
            self.retrieval_strategy.embedding_model,
            chunk_contents,
            AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )

        if progress_bar:
            progress_bar.close()
        stats_tracker.stop_timer('embedding_generation')

        # Ensure embeddings match chunks *after* potential errors in ai_embedding
        if len(all_chunks) != len(embeddings):
            logger.error(
                f"Baseline Critical error: Mismatch between chunks ({len(all_chunks)}) and embeddings ({len(embeddings)}) count after ai_embedding.")
            # This indicates a severe issue; perhaps stop or handle gracefully.
            # For now, we'll let it proceed and likely fail at zip if counts differ.

        print(f"Baseline: Start indexing embeddings...")

        # 3. Store embeddings and metadata in SQLite-Vec
        if self.sqlite_db is None:
            if os.path.exists(self.sqlite_db_file_path):
                os.remove(self.sqlite_db_file_path)
            self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
            self.sqlite_db.enable_load_extension(True)
            sqlite_vec.load(self.sqlite_db)
            self.sqlite_db.enable_load_extension(False)
            self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
            if embeddings:  # Only create table if there are embeddings
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                )
            else:  # If no embeddings (e.g. no chunks), still need embedding_infos to be an empty list
                self.embedding_infos = []
                return  # No further indexing needed

        with self.sqlite_db as db:
            insert_data = []
            # self.embedding_infos should already be initialized as an empty list
            current_rowid = 0  # Keep track of SQLite rowid (starts at 1)
            for chunk, embedding in zip(all_chunks, embeddings):  # This might fail if len mismatch
                # Add metadata to our mapping list BEFORE inserting
                self.embedding_infos.append(  # This list maps 0-based index to EmbeddingInfo
                    EmbeddingInfo(
                        document_id=chunk.file_path,
                        span=chunk.span,  # This span is of the original content part
                        processed_content=chunk.content,  # Store the full text that was embedded
                    )
                )
                insert_data.append(
                    (current_rowid + 1, serialize_f32(embedding))  # SQLite rowid for this item
                )
                current_rowid += 1

            if insert_data:  # Only execute if there's data
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )

        print(f"Baseline: Finished indexing {len(self.embedding_infos)} embeddings.")

    async def query(self, query: str) -> QueryResponse:
        if self.sqlite_db is None or self.embedding_infos is None:  # Check if initialized
            # This implies sync_all_documents might not have run or completed correctly
            logger.error("Baseline RetrievalMethod not properly synchronized. Call sync_all_documents first.")
            raise ValueError("Sync documents before querying!")

        if not self.embedding_infos:  # Check if embedding_infos is empty (e.g. no chunks were made)
            logger.info("Baseline: No embeddings available to query. Returning empty response.")
            return QueryResponse(retrieved_snippets=[])

        # 1. Get Top_K Embedding results
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
            [serialize_f32(query_embedding), self.retrieval_strategy.embedding_top_k],
        ).fetchall()

        # Get metadata using the rowid
        retrieved_metadatas: List[EmbeddingInfo] = []
        for row_id_sqlite, _ in rows:  # row_id_sqlite is 1-based
            meta_index = cast(int, row_id_sqlite) - 1  # Convert to 0-based index for self.embedding_infos
            if 0 <= meta_index < len(self.embedding_infos):
                retrieved_metadatas.append(self.embedding_infos[meta_index])
            else:
                logger.warning(
                    f"Invalid rowid {row_id_sqlite} from SQLite query mapping to embedding_infos. Max index: {len(self.embedding_infos) - 1}. Skipping.")

        # --- Prepare texts for reranking, conditionally adding summaries ---
        initial_texts_for_reranking = []
        chunk_strat_config = self.retrieval_strategy.chunking_strategy

        # Efficiently gather summaries for all unique documents in retrieved_metadatas
        doc_ids_needing_summaries = set()
        if chunk_strat_config.strategy_name.startswith("summary_") and \
                chunk_strat_config.summary_model and chunk_strat_config.summary_prompt_template:
            for meta in retrieved_metadatas:
                if meta.document_id not in self.documents:
                    logger.warning(
                        f"Document ID {meta.document_id} from EmbeddingInfo not found in self.documents. Cannot prepare text for reranking.")
                    continue
                doc_ids_needing_summaries.add(meta.document_id)

        summary_tasks = {}
        if doc_ids_needing_summaries:  # Only proceed if there are docs for which to fetch summaries
            summaries_base_dir = "./data/summaries"  # Define base dir for summary text files
            for doc_id in doc_ids_needing_summaries:
                doc_content = self.documents[doc_id].content
                summary_tasks[doc_id] = asyncio.create_task(
                    generate_document_summary(
                        document_file_path=doc_id,
                        document_content=doc_content,
                        summarization_model=chunk_strat_config.summary_model,  # type: ignore
                        summary_prompt_template=chunk_strat_config.summary_prompt_template,  # type: ignore
                        prompt_target_char_length=chunk_strat_config.prompt_target_char_length,
                        truncate_char_length=chunk_strat_config.summary_truncation_length,
                        summaries_output_dir_base=summaries_base_dir
                    )
                )

        document_summary_map: Dict[str, str] = {}
        for doc_id, task in summary_tasks.items():
            try:
                document_summary_map[doc_id] = await task
            except Exception as e:
                logger.error(
                    f"Error fetching summary for doc {doc_id} during rerank text prep: {e}. Defaulting to empty summary for this doc.")
                document_summary_map[doc_id] = ""  # Fallback

        # Construct texts for reranker
        initial_texts_for_reranking = [meta.processed_content for meta in retrieved_metadatas]

        # 2. Rerank if specified
        final_metadatas = retrieved_metadatas
        if self.retrieval_strategy.rerank_model is not None and initial_texts_for_reranking:  # Check if list not empty
            # Pass top_k=None to get all reranked indices. The truncation will happen in the benchmark script.
            reranked_indices_map = await ai_rerank(
                self.retrieval_strategy.rerank_model,
                query,
                texts=initial_texts_for_reranking,
                top_k=None,
            )
            final_metadatas = [
                retrieved_metadatas[i] for i in reranked_indices_map
            ]

        # 3. Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets: list[RetrievedSnippet] = []
        for i, metadata in enumerate(final_metadatas):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            # Use the span directly from the metadata (this is for the *original content* part)
            span = metadata.span
            text_content = self.get_embedding_info_text(metadata)  # Original content for length check
            current_len = len(text_content)
            current_full_chunk_text = metadata.processed_content

            final_span = span  # This is the span of the original content
            if remaining_tokens is not None:
                if current_len > remaining_tokens:
                    final_span = (span[0], span[0] + remaining_tokens)  # Truncate original content span
                    # current_len = remaining_tokens # This was original, but current_len refers to original_content_text
                remaining_tokens -= current_len  # Deduct based on original content length considered for snippet

            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=metadata.document_id,
                    span=final_span,  # The span is of the original content portion
                    score=1.0 / (i + 1),
                    full_chunk_text=current_full_chunk_text  # Use the stored processed_content
                )
            )

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        """Retrieves the text content for a given EmbeddingInfo."""
        # This must return the *original* content segment based on the span
        if embedding_info.document_id not in self.documents:
            logger.error(f"Document ID '{embedding_info.document_id}' not found in cached documents for EmbeddingInfo.")
            return ""
        doc_content = self.documents[embedding_info.document_id].content
        span_start, span_end = embedding_info.span
        if not (0 <= span_start <= span_end <= len(doc_content)):
            logger.error(
                f"Invalid span {embedding_info.span} for document {embedding_info.document_id} (len {len(doc_content)}).")
            return ""
        return doc_content[span_start:span_end]