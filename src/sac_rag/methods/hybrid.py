import asyncio
import os
from typing import List, Dict, Literal, Optional
import logging

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.embeddings import BaseEmbedding

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
    AIModel,
    ai_embedding,
    ai_rerank,
)
from sac_rag.utils.chunking import Chunk, get_chunks, ChunkingStrategy

logger = logging.getLogger(__name__)


# --- Configuration Model ---
class HybridStrategy(BaseModel):
    """Configuration specific to the Hybrid retrieval method."""
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    fusion_weight: float = 0.5
    rerank_model: AIRerankModel | None = None
    rerank_top_k: int | None = None
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
    print(f"fusion weight: {weights}")  # TODO: Check if this really works!!
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
    nodes: List[TextNode] | None
    vector_index: VectorStoreIndex | None
    bm25_retriever: BM25Retriever | None
    _llama_embed_model: BaseEmbedding | None = None

    def __init__(self, retrieval_strategy: HybridStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        self._llama_embed_model = None

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
            # get_chunks is now async
            task = asyncio.create_task(get_chunks(document=document, **chunking_params))  # type: ignore
            all_chunks_tasks.append(task)

        results_of_chunking_tasks = await asyncio.gather(*all_chunks_tasks)
        all_chunks: List[Chunk] = []
        for chunk_list_for_doc in results_of_chunking_tasks:
            all_chunks.extend(chunk_list_for_doc)

        print(f"Hybrid: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("Hybrid: No chunks created, skipping index creation.")
            self.nodes = []  # Ensure nodes is initialized as empty list
            return

        # 1. Fetch embeddings using ai_embedding (utilizes cache)
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

        if len(embeddings) != len(all_chunks):
            logger.error(
                f"Hybrid Critical Error: Mismatch between chunks ({len(all_chunks)}) and embeddings ({len(embeddings)}) count.")
            raise ValueError(f"Hybrid Error: Mismatch between number of chunks and obtained embeddings.")

        # 2. Create LlamaIndex TextNodes with pre-computed embeddings
        print("Hybrid: Creating LlamaIndex TextNodes with pre-computed embeddings...")
        self.nodes = []  # Initialize as list of TextNode
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            # Create a unique node ID based on file and span of original content
            node_id = f"{chunk.file_path}_{chunk.span[0]}_{chunk.span[1]}"
            node = TextNode(
                id_=node_id,
                text=chunk.content,  # This text includes the summary
                metadata={
                    "file_path": chunk.file_path,
                    # Store original content span in metadata for consistent snippet creation
                    "original_span_start": chunk.span[0],
                    "original_span_end": chunk.span[1],
                },
                # Removed start_char_idx and end_char_idx from TextNode direct attributes
                # as LlamaIndex typically infers these from text or they are less critical
                # when metadata holds the primary span. If needed by specific LlamaIndex features,
                # they would refer to spans within chunk.content (summary + original).
                # For our purposes, original_span in metadata is key.
                embedding=embedding,
            )
            self.nodes.append(node)
        print(f"Hybrid: Created {len(self.nodes)} TextNodes with embeddings.")

        # 3. Set the global LlamaIndex embedding model (likely needed for query time)
        print("Hybrid: Setting global LlamaIndex embedding model (for query time)...")
        Settings.embed_model = self._get_llama_embed_model()

        # 4. Build In-Memory Vector Index using the TextNodes with pre-computed embeddings
        print("Hybrid: Building vector index from nodes with pre-computed embeddings...")
        self.vector_index = VectorStoreIndex(  # Pass self.nodes directly
            self.nodes,  # type: ignore
            show_progress=True,  # Changed to True
        )
        print("Hybrid: Vector index built.")

        # 5. Build BM25 Retriever using the TextNodes
        print("Hybrid: Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,  # type: ignore
            similarity_top_k=self.retrieval_strategy.bm25_top_k
        )
        print("Hybrid: BM25 retriever built.")

    async def query(self, query: str) -> QueryResponse:
        """Perform Hybrid retrieval: vector + bm25 + fusion + optional reranking."""
        if self.vector_index is None or self.bm25_retriever is None or self.nodes is None:
            if self.nodes is not None and not self.nodes:
                print("Hybrid Query: No nodes available for querying (list is empty). Returning empty response.")
                return QueryResponse(retrieved_snippets=[])
            raise ValueError("Indices not synchronized. Call sync_all_documents first.")

        # 1. Get Retrievers
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.retrieval_strategy.embedding_top_k)
        bm25_retriever = self.bm25_retriever

        retrievers: List[BaseRetriever] = [vector_retriever, bm25_retriever]  # type: ignore

        # 2. Run retrievals asynchronously
        tasks = [retriever.aretrieve(query) for retriever in retrievers]  # type: ignore
        task_results: List[List[NodeWithScore]] = await asyncio.gather(*tasks)

        results_dict = {
            'vector': task_results[0] if len(task_results) > 0 else [],
            'bm25': task_results[1] if len(task_results) > 1 else [],
        }

        bm25_weight = 1 - self.retrieval_strategy.fusion_weight
        fusion_weight = {"bm25": bm25_weight, "vector": self.retrieval_strategy.fusion_weight}

        # 3. Fuse results
        fused_nodes = fuse_results_weighted_rrf(results_dict, similarity_top_k=self.retrieval_strategy.fusion_top_k, weights=fusion_weight)

        # 4. Optional Reranking Step
        final_nodes = fused_nodes
        if self.retrieval_strategy.rerank_model and self.retrieval_strategy.rerank_top_k is not None and fused_nodes:
            logger.debug(
                f"Hybrid: Reranking {len(fused_nodes)} fused nodes with {self.retrieval_strategy.rerank_model.company}/{self.retrieval_strategy.rerank_model.model} (top_k={self.retrieval_strategy.rerank_top_k})...")
            texts_to_rerank = [node.get_content() for node in fused_nodes]  # Content includes summary

            reranked_indices = await ai_rerank(
                model=self.retrieval_strategy.rerank_model,
                query=query,
                texts=texts_to_rerank,
                top_k=self.retrieval_strategy.rerank_top_k
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
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        self._llama_embed_model = None
        # Settings.embed_model = None # Optional: Reset LlamaIndex global settings
        print("Hybrid: Cleanup complete.")
