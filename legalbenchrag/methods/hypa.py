import asyncio
from typing import List, Dict, Literal

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Document as LlamaDocument, Settings
from llama_index.core.schema import NodeWithScore, BaseNode, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.embeddings import BaseEmbedding

# Import specific embedding integrations if needed for instantiation helper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.voyageai import VoyageEmbedding

from pydantic import BaseModel

from legalbenchrag.benchmark_types import (
    Document as BenchmarkDocument,
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from legalbenchrag.utils.ai import (
    AIEmbeddingModel,
    # AIEmbeddingType,
    # ai_embedding,
)
# Import new chunking utility
from legalbenchrag.utils.chunking import Chunk, get_chunks


# --- Configuration Model ---
class HypaStrategy(BaseModel):
    """Configuration specific to the HyPA retrieval method."""
    method_name: str = "hypa"
    # Add chunking strategy config to match baseline
    chunk_strategy_name: Literal["naive", "rcts"] = "rcts"
    chunk_size: int
    chunk_overlap_ratio: float = 0.0
    embedding_model: AIEmbeddingModel  # We'll pass this to LlamaIndex
    embedding_top_k: int  # Used for initial vector retrieval before fusion
    bm25_top_k: int  # Used for initial bm25 retrieval before fusion
    fusion_top_k: int  # Final number of results after fusion


# --- Helper Function for Fusion ---
def fuse_results(results_dict: Dict[str, List[NodeWithScore]], similarity_top_k: int) -> List[NodeWithScore]:
    """Fuse results using Reciprocal Rank Fusion."""
    k = 60.0  # RRF parameter
    fused_scores: Dict[str, float] = {}
    text_to_node: Dict[str, NodeWithScore] = {}

    all_nodes_with_scores: List[NodeWithScore] = []
    for nodes_with_scores_list in results_dict.values():
        all_nodes_with_scores.extend(nodes_with_scores_list)

    # Deduplicate nodes based on node_id first, keeping the highest score
    unique_nodes_by_id: Dict[str, NodeWithScore] = {}
    for node_with_score in all_nodes_with_scores:
        node_id = node_with_score.node.node_id
        if node_id not in unique_nodes_by_id or node_with_score.score > unique_nodes_by_id[node_id].score:
             unique_nodes_by_id[node_id] = node_with_score

    sorted_nodes = sorted(unique_nodes_by_id.values(), key=lambda x: x.score or 0.0, reverse=True)

    for rank, node_with_score in enumerate(sorted_nodes):
        node_id = node_with_score.node.node_id
        text_to_node[node_id] = node_with_score
        if node_id not in fused_scores:
            fused_scores[node_id] = 0.0
        fused_scores[node_id] += 1.0 / (rank + k)

    reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    reranked_nodes: List[NodeWithScore] = []
    for node_id in reranked_ids:
        node_with_score = text_to_node[node_id]
        node_with_score.score = fused_scores[node_id]  # Update score
        reranked_nodes.append(node_with_score)

    return reranked_nodes[:similarity_top_k]

# --- HyPA Retrieval Method Implementation ---
class HypaRetrievalMethod(RetrievalMethod):
    strategy: HypaStrategy
    documents: Dict[str, BenchmarkDocument]
    nodes: List[BaseNode] | None  # Store LlamaIndex TextNode objects
    vector_index: VectorStoreIndex | None
    bm25_retriever: BM25Retriever | None
    # Store the instantiated embed model for potential reuse/check
    _llama_embed_model: BaseEmbedding | None = None

    def __init__(self, strategy: HypaStrategy):
        self.strategy = strategy
        self.documents = {}
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        self._llama_embed_model = None

    def _get_llama_embed_model(self) -> BaseEmbedding:
        """Maps the strategy's AIEmbeddingModel to a LlamaIndex BaseEmbedding instance."""
        if self._llama_embed_model:
            # Simple check if model type matches, avoids re-instantiation if strategy is the same
            # Note: This doesn't deeply compare model parameters. Assumes if strategy object is same, model is.
            # A more robust check might compare self.strategy.embedding_model attributes.
            current_model_config = self.strategy.embedding_model
            if isinstance(self._llama_embed_model, OpenAIEmbedding) and current_model_config.company == 'openai' and self._llama_embed_model.model == current_model_config.model:
                return self._llama_embed_model
            if isinstance(self._llama_embed_model, HuggingFaceEmbedding) and current_model_config.company == 'huggingface' and self._llama_embed_model.model_name == current_model_config.model:
                return self._llama_embed_model
            # Add similar checks for Cohere, VoyageAI if needed

        print(f"HyPA: Instantiating LlamaIndex embedding model for: {self.strategy.embedding_model.company} / {self.strategy.embedding_model.model}")
        model_config = self.strategy.embedding_model
        embed_model: BaseEmbedding

        if model_config.company == 'openai':
            # Assuming API key is set globally via env var OPENAI_API_KEY
            embed_model = OpenAIEmbedding(model=model_config.model)
        elif model_config.company == 'huggingface':
            # Use trust_remote_code=True always as requested
            embed_model = HuggingFaceEmbedding(
                model_name=model_config.model,
                # device="cuda" # Optional: specify device, defaults to auto
                trust_remote_code=True
            )
        elif model_config.company == 'cohere':
            # Assuming API key is set globally via env var COHERE_API_KEY
             embed_model = CohereEmbedding(
                 model_name=model_config.model,
                 # input_type might be needed depending on LlamaIndex version/defaults
                 # input_type="search_document" # Or handle based on context? Usually default is okay.
             )
        elif model_config.company == 'voyageai':
             # Assuming API key is set globally via env var VOYAGEAI_API_KEY
             embed_model = VoyageEmbedding(model_name=model_config.model)
             # input_type might be needed
             # input_type="document"
        else:
            raise ValueError(f"Unsupported embedding company in HypaStrategy: {model_config.company}")

        self._llama_embed_model = embed_model
        return embed_model

    async def ingest_document(self, document: BenchmarkDocument) -> None:
        """Store document content."""
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        """Process documents, create nodes, build indices using shared chunking and configured embedder."""
        print("HyPA: Calculating chunks...")
        all_chunks: List[Chunk] = []
        for document in self.documents.values():
            chunks_for_doc = get_chunks(
                document=document,
                strategy_name=self.strategy.chunk_strategy_name,
                chunk_size=self.strategy.chunk_size,
                chunk_overlap_ratio=self.strategy.chunk_overlap_ratio,
            )
            all_chunks.extend(chunks_for_doc)
        print(f"HyPA: Created {len(all_chunks)} chunks.")

        if not all_chunks:
            print("HyPA: No chunks created, skipping index creation.")
            self.nodes = []
            return

        # 1. Set the global LlamaIndex embedding model for this strategy run
        # This needs to happen *before* index construction.
        print("HyPA: Setting global LlamaIndex embedding model...")
        Settings.embed_model = self._get_llama_embed_model()
        # Optional: Set global chunk size if needed elsewhere by LlamaIndex internals, though we provide nodes directly.
        # Settings.chunk_size = self.strategy.chunk_size

        # 2. Convert Chunks to LlamaIndex TextNodes
        print("HyPA: Creating LlamaIndex TextNodes from chunks...")
        self.nodes = []
        for i, chunk in enumerate(all_chunks):
            # Create a unique node ID based on file and span
            node_id = f"{chunk.file_path}_{chunk.span[0]}_{chunk.span[1]}"
            node = TextNode(
                id_=node_id,
                text=chunk.content,
                metadata={
                    "file_path": chunk.file_path,
                    # No need for start/end in metadata if directly on node
                },
                # Directly set start/end char indices on the node
                start_char_idx=chunk.span[0],
                end_char_idx=chunk.span[1],
                # Ensure relationships are clear if needed, default is usually okay
                # relationships={...}
            )
            self.nodes.append(node)
        print(f"HyPA: Created {len(self.nodes)} TextNodes.")

        # 3. Build In-Memory Vector Index using the TextNodes
        print("HyPA: Building vector index...")
        # Pass nodes directly, LlamaIndex will use Settings.embed_model
        self.vector_index = VectorStoreIndex(
            self.nodes,
            show_progress=True,
            # service_context=None # Deprecated, use Settings
        )
        print("HyPA: Vector index built.")

        # 4. Build BM25 Retriever using the TextNodes
        print("HyPA: Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,  # Use the same nodes
            similarity_top_k=self.strategy.bm25_top_k
        )
        print("HyPA: BM25 retriever built.")

    async def query(self, query: str) -> QueryResponse:
        """Perform HyPA retrieval: vector + bm25 + fusion."""
        if self.vector_index is None or self.bm25_retriever is None or self.nodes is None:
            # Handle case where sync didn't create nodes
            if not self.nodes:
                print("HyPA Query: No nodes available for querying. Returning empty response.")
                return QueryResponse(retrieved_snippets=[])
            raise ValueError("Indices not synchronized. Call sync_all_documents first.")

        # 1. Get Retrievers
        # Vector retriever uses the index (which used the correct embed model)
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.strategy.embedding_top_k)
        # BM25 retriever was already configured with nodes and top_k
        bm25_retriever = self.bm25_retriever
        # Re-set bm25 top_k per query if needed (usually configured at init is fine)
        # bm25_retriever.similarity_top_k = self.strategy.bm25_top_k

        retrievers: List[BaseRetriever] = [vector_retriever, bm25_retriever]

        # 2. Run retrievals asynchronously
        tasks = [retriever.aretrieve(query) for retriever in retrievers]
        task_results: List[List[NodeWithScore]] = await asyncio.gather(*tasks)

        results_dict = {
            'vector': task_results[0] if len(task_results) > 0 else [],
            'bm25': task_results[1] if len(task_results) > 1 else [],
        }

        # 3. Fuse results
        fused_nodes = fuse_results(results_dict, similarity_top_k=self.strategy.fusion_top_k)

        # 4. Map LlamaIndex Nodes to LegalBenchRAG Snippets
        retrieved_snippets: List[RetrievedSnippet] = []
        for node_with_score in fused_nodes:
            node = node_with_score.node
            # Node should be a TextNode with the info we added
            if isinstance(node, TextNode):
                file_path = node.metadata.get("file_path")
                start_idx = node.start_char_idx
                end_idx = node.end_char_idx
                score = node_with_score.score if node_with_score.score is not None else 0.0

                if file_path and start_idx is not None and end_idx is not None:
                    retrieved_snippets.append(
                        RetrievedSnippet(
                            file_path=str(file_path),
                            span=(int(start_idx), int(end_idx)),
                            score=float(score)
                        )
                    )
                else:
                    missing = []
                    if not file_path: missing.append("file_path")
                    if start_idx is None: missing.append("start_char_idx")
                    if end_idx is None: missing.append("end_char_idx")
                    print(f"HyPA WARNING: Node {node.node_id} missing required info ({', '.join(missing)}). Skipping.")
            else:
                print(f"HyPA WARNING: Retrieved node {node.node_id} is not a TextNode. Skipping.")

        # print(f"HyPA: Returning {len(retrieved_snippets)} snippets.")
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    async def cleanup(self) -> None:
        """Release resources."""
        self.documents = {}
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        self._llama_embed_model = None
        # Optionally reset LlamaIndex global settings if they interfere elsewhere
        # Settings.embed_model = None # Or reset to a default
        print("HyPA: Cleanup complete.")