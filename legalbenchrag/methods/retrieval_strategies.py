from typing import List, Union

from legalbenchrag.methods.baseline import ChunkingStrategy, RetrievalStrategy as BaselineStrategy  # BaselineStrategy are the 4 original legalbench-rag strategies
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel
from legalbenchrag.methods.hypa import HypaStrategy

# Chunking Strategies (used by both Baseline and HyPA)
chunk_strategies: List[ChunkingStrategy] = [
    ChunkingStrategy(strategy_name="rcts", chunk_size=500, chunk_overlap_ratio=0.0),
    # ChunkingStrategy(strategy_name="naive", chunk_size=500),
]

# Embedding Models (used by both)
oai_embed_model = AIEmbeddingModel(company="openai", model="text-embedding-3-large")
hf_embed_model_bge_base = AIEmbeddingModel(company="huggingface", model="BAAI/bge-base-en-v1.5")
hf_embed_model_bge_large = AIEmbeddingModel(company="huggingface", model="BAAI/bge-large-en-v1.5")
hf_embed_model_gte = AIEmbeddingModel(company="huggingface", model="thenlper/gte-large")
hf_embed_model_legalbert_base = AIEmbeddingModel(company="huggingface", model="nlpaueb/legal-bert-base-uncased")
hf_embed_model_legalbert_small = AIEmbeddingModel(company="huggingface", model="nlpaueb/legal-bert-small-uncased")

embed_strategies = [
    oai_embed_model,
    # hf_embed_model_bge_base,
    # hf_embed_model_bge_large,
    # hf_embed_model_gte,
    # hf_embed_model_legalbert_base,
    # hf_embed_model_legalbert_small,
]

# Rerank Models (used by both)
cohere_rerank_model = AIRerankModel(company="cohere", model="rerank-english-v3.0")
voyage_rerank_model = AIRerankModel(company="voyageai", model="rerank-lite-1")
hf_rerank_minilm = AIRerankModel(company="huggingface", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
hf_rerank_bge_base = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-base")
hf_rerank_bge_large = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-large")


rerank_models: list[AIRerankModel | None] = [
    None,
    # cohere_rerank_model,
    # voyage_rerank_model,
    # hf_rerank_minilm,
    hf_rerank_bge_base,
    # hf_rerank_bge_large,
]

# Define final top_k values to test for both methods
final_top_k_values: list[int] = [1, 2, 4, 8, 16, 32, 64]
# Calculate max K needed for consistent input to reranker caching
max_final_k = max(final_top_k_values) if final_top_k_values else 64  # Default if list is empty

# --- Baseline Strategy Definitions ---

BASELINE_STRATEGIES: list[BaselineStrategy] = []
for chunk_strategy in chunk_strategies:
    for embed_model in embed_strategies:
        for rerank_model in rerank_models:
            for final_k in final_top_k_values:
                # rerank_topk is final_k if reranker is active
                BASELINE_STRATEGIES.append(
                    BaselineStrategy(
                        chunking_strategy=chunk_strategy,
                        embedding_model=embed_model,
                        embedding_topk=300 if rerank_model is not None else final_k,
                        rerank_model=rerank_model,
                        rerank_topk=final_k if rerank_model is not None else 0,
                        token_limit=None,
                    ),
                )

# --- HyPA Strategy Definitions ---

HYPA_STRATEGIES: list[HypaStrategy] = []
for chunk_strategy in chunk_strategies:
    for embed_model in embed_strategies:
        for rerank_model in rerank_models:
            for final_k in final_top_k_values:
                # Determine intermediate K values based on whether reranker is active
                if rerank_model is None:
                    # No reranker: fusion_top_k is the final k
                    current_rerank_top_k = None
                    current_fusion_top_k = final_k
                    current_embedding_top_k = max(10, current_fusion_top_k * 2)
                    current_bm25_top_k = max(10, current_fusion_top_k * 2)
                else:
                    # Reranker active: final k is rerank_top_k
                    current_rerank_top_k = final_k
                    current_fusion_top_k = max(10, max_final_k * 2)
                    current_embedding_top_k = max(20, current_fusion_top_k * 3)
                    current_bm25_top_k = max(20, current_fusion_top_k * 3)

                HYPA_STRATEGIES.append(
                    HypaStrategy(
                        chunk_strategy_name=chunk_strategy.strategy_name,
                        chunk_size=chunk_strategy.chunk_size,
                        chunk_overlap_ratio=chunk_strategy.chunk_overlap_ratio,
                        embedding_model=embed_model,
                        embedding_top_k=current_embedding_top_k,
                        bm25_top_k=current_bm25_top_k,
                        fusion_top_k=current_fusion_top_k,
                        rerank_model=rerank_model,
                        rerank_top_k=current_rerank_top_k
                    )
                )


# --- Combine all strategies to be tested ---
# Use Union for type hinting the list
ALL_RETRIEVAL_STRATEGIES: List[Union[BaselineStrategy, HypaStrategy]] = []
ALL_RETRIEVAL_STRATEGIES.extend(BASELINE_STRATEGIES)
ALL_RETRIEVAL_STRATEGIES.extend(HYPA_STRATEGIES)

print(f"Defined {len(BASELINE_STRATEGIES)} Baseline strategies.")
print(f"Defined {len(HYPA_STRATEGIES)} HyPA strategies.")
print(f"Total strategies to run: {len(ALL_RETRIEVAL_STRATEGIES)}")
