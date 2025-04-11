import asyncio
from typing import List, Dict

from llama_index.core import VectorStoreIndex, Document as LlamaDocument  # Rename to avoid clash
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, BaseNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from pydantic import BaseModel

from legalbenchrag.benchmark_types import (
    Document as BenchmarkDocument,  # Rename to avoid clash
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from legalbenchrag.utils.ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    ai_embedding,  # We might not need direct ai_embedding if LlamaIndex handles it
)
# Import necessary LlamaIndex components if not already imported via other modules
from llama_index.core import Settings


# --- Configuration Model ---
class HypaStrategy(BaseModel):
    """Configuration specific to the HyPA retrieval method."""
    method_name: str = "hypa"  # Identifier
    chunk_size: int
    embedding_model: AIEmbeddingModel  # We'll pass this to LlamaIndex
    embedding_top_k: int  # Used for initial vector retrieval before fusion
    bm25_top_k: int  # Used for initial bm25 retrieval before fusion
    fusion_top_k: int  # Final number of results after fusion


# --- Helper Function for Fusion (Adapted from HyPA-RAG code) ---
def fuse_results(results_dict: Dict, similarity_top_k: int) -> List[NodeWithScore]:
    """Fuse results using Reciprocal Rank Fusion."""
    k = 60.0  # RRF parameter
    fused_scores: Dict[str, float] = {}
    text_to_node: Dict[str, NodeWithScore] = {}

    all_nodes_with_scores: List[NodeWithScore] = []
    for nodes_with_scores_list in results_dict.values():
         all_nodes_with_scores.extend(nodes_with_scores_list)

    # Compute reciprocal rank scores
    # Note: This assumes results_dict contains lists of NodeWithScore from different retrievers for potentially the *same* query concepts
    # A simple adaptation is to process all nodes together, giving rank based on original retriever score
    # A more complex RRF would require ranking within each sub-query result list first.
    # For simplicity now, we'll rank all retrieved nodes globally based on their initial score.

    # Deduplicate nodes based on node_id first, keeping the highest score
    unique_nodes_by_id: Dict[str, NodeWithScore] = {}
    for node_with_score in all_nodes_with_scores:
        node_id = node_with_score.node.node_id
        if node_id not in unique_nodes_by_id or node_with_score.score > unique_nodes_by_id[node_id].score:
             unique_nodes_by_id[node_id] = node_with_score

    sorted_nodes = sorted(unique_nodes_by_id.values(), key=lambda x: x.score or 0.0, reverse=True)

    for rank, node_with_score in enumerate(sorted_nodes):
        node_id = node_with_score.node.node_id # Use node_id for uniqueness
        # Store the node itself for later retrieval
        text_to_node[node_id] = node_with_score # Map id to node
        if node_id not in fused_scores:
            fused_scores[node_id] = 0.0
        # Add reciprocal rank score
        fused_scores[node_id] += 1.0 / (rank + k)

    # Sort results by fused score
    reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    # Create the final list of nodes with fused scores
    reranked_nodes: List[NodeWithScore] = []
    for node_id in reranked_ids:
        node_with_score = text_to_node[node_id]
        node_with_score.score = fused_scores[node_id] # Update score to fused score
        reranked_nodes.append(node_with_score)

    return reranked_nodes[:similarity_top_k]

# --- HyPA Retrieval Method Implementation ---
class HypaRetrievalMethod(RetrievalMethod):
    strategy: HypaStrategy
    documents: Dict[str, BenchmarkDocument] # Store original documents
    nodes: List[BaseNode] | None # Store processed LlamaIndex nodes
    vector_index: VectorStoreIndex | None
    bm25_retriever: BM25Retriever | None

    def __init__(self, strategy: HypaStrategy):
        self.strategy = strategy
        self.documents = {}
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        # Configure LlamaIndex Settings (optional, can be done globally too)
        # Settings.llm = ... # If HyPA needs specific LLM calls not covered by ai.py
        # Settings.embed_model = ... # How to bridge ai_embedding with LlamaIndex?
        # LlamaIndex typically takes the embedding model object directly.
        # We might need a small wrapper if ai_embedding is just a function.
        # For now, assume we can instantiate the HuggingFaceEmbedding model here or globally.
        # Easiest is likely setting Settings.embed_model globally before benchmark run.

    async def ingest_document(self, document: BenchmarkDocument) -> None:
        """Store document content."""
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        """Process documents, create nodes, build indices."""
        print("HyPA: Processing documents and building indices...")
        llama_docs: List[LlamaDocument] = []
        for file_path, doc in self.documents.items():
             # Create LlamaIndex Document, ensuring metadata includes file_path
             # LlamaDocument automatically tracks metadata like file_path if loaded via readers,
             # here we manually construct it.
             llama_docs.append(LlamaDocument(text=doc.content, metadata={"file_path": file_path}))

        # 1. Create Nodes using SentenceSplitter
        # We rely on the splitter to automatically add 'start_char_idx', 'end_char_idx' metadata if possible,
        # and it inherits the 'file_path' from the LlamaDocument.
        splitter = SentenceSplitter(chunk_size=self.strategy.chunk_size, chunk_overlap=int(self.strategy.chunk_size * 0.1)) # Small overlap
        self.nodes = splitter.get_nodes_from_documents(llama_docs, show_progress=True)
        print(f"HyPA: Created {len(self.nodes)} nodes.")

        if not self.nodes:
             print("HyPA: No nodes created, skipping index creation.")
             return

        # Verify metadata (optional check)
        # for node in self.nodes[:2]:
        #     print(f"Node Metadata Check: {node.metadata}")

        # 2. Build In-Memory Vector Index
        # Assumes Settings.embed_model is configured correctly beforehand
        print("HyPA: Building vector index...")
        self.vector_index = VectorStoreIndex(self.nodes, show_progress=True)
        print("HyPA: Vector index built.")

        # 3. Build BM25 Retriever
        print("HyPA: Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=self.strategy.bm25_top_k)
        print("HyPA: BM25 retriever built.")

    async def query(self, query: str) -> QueryResponse:
        """Perform HyPA retrieval: vector + bm25 + fusion."""
        if self.vector_index is None or self.bm25_retriever is None or self.nodes is None:
            raise ValueError("Indices not synchronized. Call sync_all_documents first.")

        # print(f"HyPA: Received query: {query}")

        # 1. Define Retrievers for this query
        vector_retriever = self.vector_index.as_retriever(similarity_top_k=self.strategy.embedding_top_k)
        # BM25 retriever needs similarity_top_k potentially set per-query if different from initialization
        # For simplicity, we assume the top_k set during init is sufficient for the initial fetch.
        # If bm25_retriever allowed direct top_k override in retrieve, we'd use strategy.bm25_top_k here.

        retrievers: List[BaseRetriever] = [vector_retriever, self.bm25_retriever]

        # 2. Run retrievals asynchronously (mimic run_queries)
        tasks = []
        for retriever in retrievers:
             tasks.append(retriever.aretrieve(query))

        task_results: List[List[NodeWithScore]] = await asyncio.gather(*tasks)

        # Prepare results dict for fusion function
        results_dict = {}
        # task_results[0] corresponds to vector_retriever, task_results[1] to bm25
        if len(task_results) > 0:
            results_dict['vector'] = task_results[0]
        if len(task_results) > 1:
            results_dict['bm25'] = task_results[1]

        # print(f"HyPA: Vector Results Count: {len(results_dict.get('vector', []))}")
        # print(f"HyPA: BM25 Results Count: {len(results_dict.get('bm25', []))}")


        # 3. Fuse results
        fused_nodes = fuse_results(results_dict, similarity_top_k=self.strategy.fusion_top_k)
        # print(f"HyPA: Fused Results Count: {len(fused_nodes)}")

        # 4. Map LlamaIndex Nodes to LegalBenchRAG Snippets
        retrieved_snippets: List[RetrievedSnippet] = []
        for node_with_score in fused_nodes:
            node = node_with_score.node
            metadata = node.metadata
            file_path = metadata.get("file_path")
            # Get indices directly from node attributes
            # start_idx = metadata.get("start_char_idx")
            start_idx = node.start_char_idx
            # end_idx = metadata.get("end_char_idx")
            end_idx = node.end_char_idx
            score = node_with_score.score if node_with_score.score is not None else 0.0

            if file_path is not None and start_idx is not None and end_idx is not None:
                retrieved_snippets.append(
                    RetrievedSnippet(
                        file_path=str(file_path),  # Ensure string
                        span=(int(start_idx), int(end_idx)),  # Ensure int
                        score=float(score)  # Ensure float
                    )
                )
            else:
                # Refined warning message
                missing_info = []
                if file_path is None: missing_info.append("file_path (from metadata)")
                if start_idx is None: missing_info.append("start_char_idx (from node attribute)")
                if end_idx is None: missing_info.append("end_char_idx (from node attribute)")
                print(f"!!!!!!!!!!! HyPA WARNING: Node {node.node_id} missing required info ({', '.join(missing_info)}). Skipping.")
                # print(f"Problematic Node Metadata: {metadata}")
                # print(f"Problematic Node Attributes: start={start_idx}, end={end_idx}")

        print(f"HyPA: Returning {len(retrieved_snippets)} snippets.")
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    async def cleanup(self) -> None:
        """Release resources (optional for in-memory)."""
        self.documents = {}
        self.nodes = None
        self.vector_index = None
        self.bm25_retriever = None
        print("HyPA: Cleanup complete.")