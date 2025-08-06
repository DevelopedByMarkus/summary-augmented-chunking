import asyncio
import os
import hashlib
from collections import defaultdict
from typing import List, Dict
import logging

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.embeddings import BaseEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from sac_rag.data_models import (
    Document as BenchmarkDocument,
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
)
from sac_rag.utils.chunking import Chunk, get_chunks, ChunkingStrategy
from sac_rag.utils.stats_tracker import stats_tracker
from sac_rag.utils.abbreviations import ABBREVIATIONS

logger = logging.getLogger(__name__)


# --- Configuration Model ---
class HybridStrategy(BaseModel):
    """Configuration specific to the Hybrid retrieval method."""
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    fusion_weight: float
    rerank_model: AIRerankModel | None
    rerank_top_k: List[int]
    token_limit: int | None  # TODO: Not used yet.


# --- Helper Function for Fusion ---
def fuse_results_weighted_rrf(
        results_dict: Dict[str, List[NodeWithScore]],
        similarity_top_k: int,
        weights: Dict[str, float] = None,
        k: float = 60.0,
) -> List[NodeWithScore]:
    """
    Fuses results from multiple retrievers using Weighted Reciprocal Rank Fusion.

    Args:
        results_dict: A dictionary where keys are retriever names (e.g., "bm25", "vector")
                      and values are lists of ranked NodeWithScore.
        similarity_top_k: The number of top results to return.
        weights: A dictionary mapping retriever names to their fusion weight.
                 The weights should ideally sum to 1.
        k: The ranking constant for RRF (default is 60).
    """
    if weights is None:
        weights = {"bm25": 0.0, "vector": 1.0}  # Default weights for bm25 and vector retrievers
    logger.debug(f"fusion weight: {weights}")
    # Ensure that all retrievers in the results have a corresponding weight
    if not all(key in weights for key in results_dict.keys()):
        raise ValueError(
            "The 'weights' dictionary must contain a key for each retriever in 'results_dict'."
        )

    fused_scores: Dict[str, float] = {}
    text_to_node: Dict[str, NodeWithScore] = {}

    # Iterate through each retriever's result list
    for retriever_name, nodes_with_scores in results_dict.items():
        retriever_weight = weights.get(retriever_name, 0)
        if retriever_weight == 0:
            continue

        # For each document in the list, calculate its weighted reciprocal rank
        for rank, node_with_score in enumerate(nodes_with_scores):
            node_id = node_with_score.node.node_id
            if not node_id:
                continue

            # If we see this node for the first time, add it to our mapping
            if node_id not in text_to_node:
                text_to_node[node_id] = node_with_score

            # Calculate the weighted RRF score and add it to the existing score
            # for that node. The .get() method handles the first time we see a node.
            current_score = fused_scores.get(node_id, 0.0)
            fused_scores[node_id] = current_score + retriever_weight * (1.0 / (k + rank))

    # Sort the nodes based on their final fused scores in descending order
    reranked_ids = sorted(
        fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True
    )

    # Build the final list of NodeWithScore objects
    reranked_nodes: List[NodeWithScore] = []
    for node_id in reranked_ids:
        # Retrieve the original NodeWithScore object
        node_with_score = text_to_node[node_id]
        # Update its score to the new fused score
        node_with_score.score = fused_scores[node_id]
        reranked_nodes.append(node_with_score)

    return reranked_nodes[:similarity_top_k]


# --- Hybrid Retrieval Method Implementation ---
class HybridRetrievalMethod(RetrievalMethod):
    retrieval_strategy: HybridStrategy
    documents: Dict[str, BenchmarkDocument]
    bm25_retriever: BM25Retriever | None
    _llama_embed_model: BaseEmbedding | None = None
    vector_store: ChromaVectorStore
    storage_context: StorageContext
    vector_index: VectorStoreIndex | None

    def __init__(self, retrieval_strategy: HybridStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.bm25_retriever = None
        self._llama_embed_model = None
        self.vector_index = None

        # Create a persistent client
        db = chromadb.PersistentClient(path="./data/cache/hybrid_chroma_db")

        # Get or create a collection (like a table in a database)
        # We can name it based on key strategy parameters to avoid conflicts
        collection_name = self._get_unique_collection_name(retrieval_strategy)

        print(f"Hybrid: Using ChromaDB collection: {collection_name}")
        chroma_collection = db.get_or_create_collection(collection_name)

        # Assign the vector store to the instance
        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    def _get_unique_collection_name(self, strategy: HybridStrategy) -> str:
        """
        Creates a unique, short, and valid collection name from the strategy config
        by using predefined abbreviations.
        """
        # Use .get(key, key) as a fallback to just use the full name if an abbreviation is missing.

        # 1. Embedding Model Part
        embed_model = strategy.embedding_model
        embed_abbr = ABBREVIATIONS["embedding"].get(embed_model.model, embed_model.model)
        embed_part = f"e_{embed_abbr}"

        # 2. Chunking Strategy Part
        cs = strategy.chunking_strategy
        chunk_abbr = ABBREVIATIONS["chunking"].get(cs.strategy_name, cs.strategy_name)
        chunk_part = f"c_{chunk_abbr}_{cs.chunk_size}_{str(cs.chunk_overlap_ratio).replace('.', 'p')}"

        # 3. Summarization Part (if applicable)
        summary_part = ""
        if cs.strategy_name.startswith("summary_") and cs.summary_model and cs.summary_prompt_template:
            # Hash the long prompt template to get a short, fixed-length ID
            prompt_hash = hashlib.sha256(cs.summary_prompt_template.encode()).hexdigest()[:8]

            s_model = cs.summary_model
            s_model_abbr = ABBREVIATIONS["embedding"].get(s_model.model, s_model.model)  # Reuse embedding dict
            summary_model_part = f"s_{s_model_abbr}"
            summary_params_part = f"{cs.prompt_target_char_length}_{cs.summary_truncation_length}_{prompt_hash}"
            summary_part = f"{summary_model_part}_{summary_params_part}"

        # 4. Combine and Sanitize
        # Combine all parts into a single string
        full_name = f"{embed_part}_{chunk_part}_{summary_part}"
        # Sanitize for ChromaDB rules: replace invalid chars
        sanitized_name = full_name.replace("/", "_").replace("-", "_").replace(".", "_")

        # Enforce length limit (3-63 chars)
        # We trim from the middle if too long, preserving the start and end
        if len(sanitized_name) > 63:
            sanitized_name = sanitized_name[:31] + ".." + sanitized_name[-30:]

        return sanitized_name.lower()

    def _get_llama_embed_model(self) -> BaseEmbedding:
        """Maps the strategy's AIEmbeddingModel to a LlamaIndex BaseEmbedding instance."""
        if self._llama_embed_model:
            current_model_config = self.retrieval_strategy.embedding_model
            if isinstance(self._llama_embed_model,
                          OpenAIEmbedding) and current_model_config.company == 'openai' and self._llama_embed_model.model_name == current_model_config.model:  # Changed .model to .model_name
                return self._llama_embed_model
            if isinstance(self._llama_embed_model,
                          HuggingFaceEmbedding) and current_model_config.company == 'huggingface' and self._llama_embed_model.model_name == current_model_config.model:
                return self._llama_embed_model
            if isinstance(self._llama_embed_model,
                          CohereEmbedding) and current_model_config.company == 'cohere' and self._llama_embed_model.model_name == current_model_config.model:  # Changed .model to .model_name
                return self._llama_embed_model
            if isinstance(self._llama_embed_model,
                          VoyageEmbedding) and current_model_config.company == 'voyageai' and self._llama_embed_model.model_name == current_model_config.model:  # Changed .model to .model_name
                return self._llama_embed_model

        print(
            f"Hybrid: Instantiating LlamaIndex embedding model for: {self.retrieval_strategy.embedding_model.company} / {self.retrieval_strategy.embedding_model.model}")
        model_config = self.retrieval_strategy.embedding_model
        embed_model: BaseEmbedding

        if model_config.company == 'openai':
            embed_model = OpenAIEmbedding(model=model_config.model, api_key=os.getenv("OPENAI_API_KEY"))
        elif model_config.company == 'huggingface':
            embed_model = HuggingFaceEmbedding(
                model_name=model_config.model,
                trust_remote_code=True
            )
        elif model_config.company == 'cohere':
            embed_model = CohereEmbedding(
                model_name=model_config.model,
                cohere_api_key=os.getenv("COHERE_API_KEY")
            )
        elif model_config.company == 'voyageai':
            embed_model = VoyageEmbedding(model_name=model_config.model, voyage_api_key=os.getenv("VOYAGEAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported embedding company in HybridStrategy: {model_config.company}")

        self._llama_embed_model = embed_model
        return embed_model

    async def ingest_document(self, document: BenchmarkDocument) -> None:
        """Store document content."""
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        """Process documents, create nodes with cached embeddings, build indices."""

        stats_tracker.start_timer('chunking_and_summarization')
        print(f"Hybrid: Calculating chunks using strategy '{self.retrieval_strategy.chunking_strategy.strategy_name}'...")

        # Prepare kwargs for get_chunks
        chunking_params = {
            "strategy_name": self.retrieval_strategy.chunking_strategy.strategy_name,
            "chunk_size": self.retrieval_strategy.chunking_strategy.chunk_size,  # Total size for summary strats
            "chunk_overlap_ratio": self.retrieval_strategy.chunking_strategy.chunk_overlap_ratio,
        }
        if self.retrieval_strategy.chunking_strategy.strategy_name.startswith("summary_"):
            chunking_params["summarization_model"] = self.retrieval_strategy.chunking_strategy.summary_model
            chunking_params["summary_prompt_template"] = self.retrieval_strategy.chunking_strategy.summary_prompt_template
            chunking_params["prompt_target_char_length"] = self.retrieval_strategy.chunking_strategy.prompt_target_char_length
            chunking_params["summary_truncation_length"] = self.retrieval_strategy.chunking_strategy.summary_truncation_length

        all_chunks_tasks: List[asyncio.Task[List[Chunk]]] = []
        for document in self.documents.values():
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)
        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        stats_tracker.set('chunks_created', len(all_chunks))
        stats_tracker.stop_timer('chunking_and_summarization')
        print(f"Hybrid: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("Hybrid: No chunks created, skipping index creation.")
            return

        # 1. Fetch embeddings using ai_embedding (utilizes cache)
        stats_tracker.start_timer('embedding_generation')
        chunk_contents = [chunk.content for chunk in all_chunks]  # These contents include summaries
        model_config = self.retrieval_strategy.embedding_model

        pbar = tqdm(total=len(chunk_contents), desc="Hybrid Embeddings", ncols=100)

        def progress_callback():
            pbar.update(1)

        embeddings = await ai_embedding(
            model=model_config,
            texts=chunk_contents,
            embedding_type=AIEmbeddingType.DOCUMENT,
            callback=progress_callback,
        )
        pbar.close()
        stats_tracker.stop_timer('embedding_generation')

        if len(embeddings) != len(all_chunks):
            logger.error(
                f"Hybrid Critical Error: Mismatch between chunks ({len(all_chunks)}) and embeddings ({len(embeddings)}) count.")
            raise ValueError(f"Hybrid Error: Mismatch between number of chunks and obtained embeddings.")

        # 2. Create LlamaIndex TextNodes and INGEST IN BATCHES
        print("Hybrid: Ingesting nodes into ChromaDB in batches...")

        batch_size = 512
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="Ingesting to ChromaDB"):
            batch_chunks = all_chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_nodes = []

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                node_id = f"{chunk.file_path}_{chunk.span[0]}_{chunk.span[1]}"
                node = TextNode(
                    id_=node_id,
                    text=chunk.content,
                    metadata={
                        "file_path": chunk.file_path,
                        "original_span_start": chunk.span[0],
                        "original_span_end": chunk.span[1],
                    },
                    embedding=embedding,
                )
                batch_nodes.append(node)

            # Add the batch of nodes to the persistent vector store
            await self.vector_store.async_add(batch_nodes)

        print(f"Hybrid: Finished ingesting {len(all_chunks)} nodes into ChromaDB.")

        # 3. Set the global LlamaIndex embedding model (likely needed for query time)
        print("Hybrid: Setting global LlamaIndex embedding model (for query time)...")
        Settings.embed_model = self._get_llama_embed_model()

        # 4. Build Vector Index from the persistent store
        print("Hybrid: Building vector index from persistent ChromaDB store...")
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )
        print("Hybrid: Vector index built.")

        # 5. Build BM25 Retriever by loading nodes from the index's docstore
        print("Hybrid: Building BM25 retriever...")
        # This loads nodes from the ChromaDB docstore
        nodes_for_bm25 = list(self.vector_index.docstore.docs.values())
        if not nodes_for_bm25:
            print("Hybrid: WARNING: No nodes found in the docstore to build BM25 index.")
            self.bm25_retriever = None
        else:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes_for_bm25,
                similarity_top_k=self.retrieval_strategy.bm25_top_k
            )
        print("Hybrid: BM25 retriever built.")

    async def query(self, query: str) -> QueryResponse:
        """Perform Hybrid retrieval: vector + bm25 + fusion + optional reranking."""
        if self.vector_index is None:
            if self.vector_store.client.count() == 0:
                print("Hybrid Query: No nodes available for querying (store is empty). Returning empty response.")
                return QueryResponse(retrieved_snippets=[])
            raise ValueError("Indices not synchronized. Call sync_all_documents first.")

        bm25_weight = 1 - self.retrieval_strategy.fusion_weight
        fusion_weight = {"bm25": bm25_weight, "vector": self.retrieval_strategy.fusion_weight}

        tasks = []
        retriever_names = []

        if fusion_weight['vector'] > 0.0:
            vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.retrieval_strategy.embedding_top_k)
            tasks.append(vector_retriever.aretrieve(query))
            retriever_names.append('vector')

        if fusion_weight['bm25'] > 0.0 and self.bm25_retriever is not None:
            tasks.append(self.bm25_retriever.aretrieve(query))
            retriever_names.append('bm25')

        # 2. Run only the necessary retrievals asynchronously.
        if tasks:
            task_results = await asyncio.gather(*tasks)
        else:  # Handle edge case where both weights are 0
            task_results = []

        # 3. Safely build the results' dictionary.
        results_dict = defaultdict(list)
        for name, result in zip(retriever_names, task_results):
            results_dict[name] = result

        # 4. Fuse results.
        fused_nodes = fuse_results_weighted_rrf(
            results_dict, self.retrieval_strategy.fusion_top_k, fusion_weight
        )

        # 4. Optional Reranking Step
        final_nodes = fused_nodes
        if self.retrieval_strategy.rerank_model is not None and fused_nodes:
            logger.debug(
                f"Hybrid: Reranking {len(fused_nodes)} fused nodes with {self.retrieval_strategy.rerank_model.company}/{self.retrieval_strategy.rerank_model.model} (top_k={self.retrieval_strategy.rerank_top_k})...")
            texts_to_rerank = [node.get_content() for node in fused_nodes]  # Content includes summary

            reranked_indices = await ai_rerank(
                model=self.retrieval_strategy.rerank_model,
                query=query,
                texts=texts_to_rerank,
                top_k=None
            )
            final_nodes = [fused_nodes[i] for i in reranked_indices]
            # Update scores to reflect reranking order
            for rank, node_with_score in enumerate(final_nodes):
                node_with_score.score = 1.0 / (rank + 1.0)  # Simple rank-based score

        # 5. Map Final LlamaIndex Nodes to LegalBenchRAG Snippets
        retrieved_snippets: List[RetrievedSnippet] = []
        for node_with_score in final_nodes:
            node = node_with_score.node
            if isinstance(node, TextNode):  # Ensure it's a TextNode
                file_path = node.metadata.get("file_path")
                # Retrieve the original content span from metadata
                original_span_start = node.metadata.get("original_span_start")
                original_span_end = node.metadata.get("original_span_end")
                score = node_with_score.score if node_with_score.score is not None else 0.0
                current_full_chunk_text = node.get_content()

                if file_path and original_span_start is not None and original_span_end is not None:
                    retrieved_snippets.append(
                        RetrievedSnippet(
                            file_path=str(file_path),
                            span=(int(original_span_start), int(original_span_end)),  # Use original span
                            score=float(score),
                            full_chunk_text=current_full_chunk_text  # Span conditionally including summary
                        )
                    )
                else:  # Log missing metadata more informatively
                    missing_meta = [item for item in ["file_path", "original_span_start", "original_span_end"] if
                                    node.metadata.get(item) is None]
                    logger.warning(
                        f"Hybrid WARNING: Node {node.node_id} missing metadata: {', '.join(missing_meta)}. Skipping.")
            else:
                logger.warning(
                    f"Hybrid WARNING: Retrieved node {node.node_id} is not a TextNode but {type(node)}. Skipping.")

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    async def cleanup(self) -> None:
        """Release resources."""
        self.documents = {}
        self.vector_index = None
        self.bm25_retriever = None
        self._llama_embed_model = None
        print("Hybrid: Cleanup complete.")
