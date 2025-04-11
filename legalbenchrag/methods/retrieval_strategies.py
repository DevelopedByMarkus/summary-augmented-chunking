from typing import Literal

from legalbenchrag.methods.baseline import ChunkingStrategy, RetrievalStrategy as BaselineStrategy  # BaselineStrategy are the 4 original legalbench-rag strategies
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel
from legalbenchrag.methods.hypa import HypaStrategy

chunk_strategy_names: list[Literal["naive", "rcts"]] = ["naive"]  # ["naive", "rcts"] MR
rerank_models: list[AIRerankModel | None] = [
    None,
    # AIRerankModel(company="cohere", model="rerank-english-v3.0"), # MR: activate if rerank is desired
]
chunk_sizes: list[int] = [500]
top_ks: list[int] = [1]  # MR [1, 2, 4, 8, 16, 32, 64]

BASELINE_STRATEGIES: list[BaselineStrategy] = []
for chunk_strategy_name in chunk_strategy_names:
    for chunk_size in chunk_sizes:
        chunking_strategy = ChunkingStrategy(
            strategy_name=chunk_strategy_name,
            chunk_size=chunk_size,
        )
        for rerank_model in rerank_models:
            for top_k in top_ks:
                BASELINE_STRATEGIES.append(
                    BaselineStrategy(
                        chunking_strategy=chunking_strategy,
                        embedding_model=AIEmbeddingModel(
                            company="openai",
                            model="text-embedding-3-large",
                        ),
                        embedding_topk=300 if rerank_model is not None else top_k,
                        rerank_model=rerank_model,
                        rerank_topk=top_k,
                        token_limit=None,
                    ),
                )

# --- Define HyPA Strategy Instances ---
HYPA_STRATEGIES: list[HypaStrategy] = []

# Example: Add one minimal Hypa strategy
HYPA_STRATEGIES.append(
    HypaStrategy(
        # method_name is implicitly "hypa"
        chunk_size=500,  # A common chunk size used in HyPA examples
        embedding_model=AIEmbeddingModel(  # Use the same embedding model for comparison
            company="openai",
            model="text-embedding-3-large",
        ),
        embedding_top_k=5,  # Retrieve more initially for vector
        bm25_top_k=5,      # Retrieve more initially for bm25
        fusion_top_k=1     # Final desired number of snippets after fusion (minimal for test)
    )
)

# --- Combine all strategies to be tested ---
ALL_RETRIEVAL_STRATEGIES: list[BaselineStrategy | HypaStrategy] = []
ALL_RETRIEVAL_STRATEGIES.extend(BASELINE_STRATEGIES)
ALL_RETRIEVAL_STRATEGIES.extend(HYPA_STRATEGIES)