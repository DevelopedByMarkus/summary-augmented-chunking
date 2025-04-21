from typing import Literal, List, Union

from legalbenchrag.methods.baseline import ChunkingStrategy as BaselineChunkingStrategy, RetrievalStrategy as BaselineStrategy  # BaselineStrategy are the 4 original legalbench-rag strategies
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel
from legalbenchrag.methods.hypa import HypaStrategy

# --- Baseline Strategy Definitions ---

# Define common chunking strategies for baseline
chunk_strategies: List[BaselineChunkingStrategy] = [
    BaselineChunkingStrategy(strategy_name="rcts", chunk_size=500, chunk_overlap_ratio=0.0),
    # BaselineChunkingStrategy(strategy_name="naive", chunk_size=500),
]

# Define common embedding models
oai_embed_model = AIEmbeddingModel(company="openai", model="text-embedding-3-large")
hf_embed_model_bge = AIEmbeddingModel(company="huggingface", model="BAAI/bge-base-en-v1.5")
hf_embed_model_gte = AIEmbeddingModel(company="huggingface", model="thenlper/gte-large")

baseline_embed_strategies = [
    # hf_embed_model_bge,
    oai_embed_model,
]

# Define rerank models
rerank_models: list[AIRerankModel | None] = [
    None,
    # AIRerankModel(company="cohere", model="rerank-english-v3.0"),
]

# Define top_k values
top_ks: list[int] = [1]  # , 2, 4, 8, 16, 32, 64]

BASELINE_STRATEGIES: list[BaselineStrategy] = []
for chunk_strategy in chunk_strategies:
    for embed_model in baseline_embed_strategies:
        for rerank_model in rerank_models:
            for top_k in top_ks:
                # Adjust embedding_topk if reranking is used
                BASELINE_STRATEGIES.append(
                    BaselineStrategy(
                        chunking_strategy=chunk_strategy,
                        embedding_model=embed_model,
                        embedding_topk=300 if rerank_model is not None else top_k,  # Fetch more initially for reranking
                        rerank_model=rerank_model,
                        # Use the final top_k for rerank_topk if reranker is active
                        rerank_topk=top_k if rerank_model is not None else 0,  # Rerank topk only relevant if reranker exists
                        token_limit=None,
                    ),
                )

# --- Define HyPA Strategy Instances ---
HYPA_STRATEGIES: list[HypaStrategy] = []

# Example HyPA strategies using different chunking and embeddings
hypa_top_ks = [1, 2, 4, 8, 16, 32, 64]  # Final fusion top_k

hypa_embed_strategies = [
    hf_embed_model_bge,
    # oai_embed_model,
]

for chunk_strategy in chunk_strategies:
    for embed_model in hypa_embed_strategies:
        for fusion_k in hypa_top_ks:
            HYPA_STRATEGIES.append(
                HypaStrategy(
                    chunk_strategy_name=chunk_strategy.strategy_name,
                    chunk_size=chunk_strategy.chunk_size,
                    chunk_overlap_ratio=chunk_strategy.chunk_overlap_ratio,
                    embedding_model=embed_model,
                    embedding_top_k=max(10, fusion_k * 2),  # Fetch more for vector search (e.g., 10 or 2x final k)
                    bm25_top_k=max(10, fusion_k * 2),  # Fetch more for BM25 (e.g., 10 or 2x final k)
                    fusion_top_k=fusion_k
                )
            )


# --- Combine all strategies to be tested ---
# Use Union for type hinting the list
ALL_RETRIEVAL_STRATEGIES: List[Union[BaselineStrategy, HypaStrategy]] = []
ALL_RETRIEVAL_STRATEGIES.extend(BASELINE_STRATEGIES)
# ALL_RETRIEVAL_STRATEGIES.extend(HYPA_STRATEGIES)

print(f"Defined {len(BASELINE_STRATEGIES)} Baseline strategies.")
print(f"Defined {len(HYPA_STRATEGIES)} HyPA strategies.")
print(f"Total strategies to run: {len(ALL_RETRIEVAL_STRATEGIES)}")
