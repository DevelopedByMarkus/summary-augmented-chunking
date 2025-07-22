from typing import List, Union

from src.sac_rag.methods.baseline import ChunkingStrategy, RetrievalStrategy as BaselineStrategy
from src.sac_rag.utils.ai import AIEmbeddingModel, AIRerankModel, AIModel
from src.sac_rag.methods.hybrid import HybridStrategy

# --- Default Summarization Settings ---
DEFAULT_SUMMARIZATION_MODEL = AIModel(company="openai", model="gpt-4o-mini")
# DEFAULT_SUMMARIZATION_MODEL = AIModel(company="huggingface", model="meta-llama/Llama-3-8B-Instruct")
DEFAULT_SUMMARY_PROMPT_TEMPLATE = """System: You are an expert legal document summarizer.
User: Summarize the following legal document text. Focus on extracting the most important entities, core purpose, and key legal topics. The summary must be concise, maximum {target_char_length} characters long, and optimized for providing context to smaller text chunks. Output only the summary text. Document:
{document_content}"""
DEFAULT_PROMPT_TARGET_CHAR_LENGTH = 200  # 150  # Target for LLM's generation
DEFAULT_SUMMARY_TRUNCATION_LENGTH = 220  # 170  # Hard truncation limit after generation

# Chunking Strategies (used by both Baseline and HyPA)
chunk_strategies: List[ChunkingStrategy] = [
    ChunkingStrategy(strategy_name="rcts", chunk_size=500, chunk_overlap_ratio=0.0),
    ChunkingStrategy(strategy_name="summary_rcts", chunk_size=500, chunk_overlap_ratio=0.0, summary_model=DEFAULT_SUMMARIZATION_MODEL, summary_prompt_template=DEFAULT_SUMMARY_PROMPT_TEMPLATE, prompt_target_char_length=DEFAULT_PROMPT_TARGET_CHAR_LENGTH, summary_truncation_length=DEFAULT_SUMMARY_TRUNCATION_LENGTH),
    # ChunkingStrategy(strategy_name="naive", chunk_size=500),
    # ChunkingStrategy(strategy_name="summary_naive", chunk_size=500, summary_model=DEFAULT_SUMMARIZATION_MODEL, summary_prompt_template=DEFAULT_SUMMARY_PROMPT_TEMPLATE, prompt_target_char_length=DEFAULT_PROMPT_TARGET_CHAR_LENGTH, summary_truncation_length=DEFAULT_SUMMARY_TRUNCATION_LENGTH),
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
voyage_rerank_model = AIRerankModel(company="voyageai", model="rerank-2-lite")
hf_rerank_minilm = AIRerankModel(company="huggingface", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
hf_rerank_bge_base = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-base")
hf_rerank_bge_large = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-large")


rerank_models: list[AIRerankModel | None] = [
    None,
    # cohere_rerank_model,
    # voyage_rerank_model,
    # hf_rerank_minilm,
    # hf_rerank_bge_base,
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
                # rerank_top_k is final_k if reranker is active
                BASELINE_STRATEGIES.append(
                    BaselineStrategy(
                        chunking_strategy=chunk_strategy,
                        embedding_model=embed_model,
                        embedding_top_k=300 if rerank_model is not None else final_k,
                        rerank_model=rerank_model,
                        rerank_top_k=final_k,
                        token_limit=None,
                    ),
                )

# --- HyPA Strategy Definitions ---

HYPA_STRATEGIES: list[HybridStrategy] = []
for chunk_strategy in chunk_strategies:
    for embed_model in embed_strategies:
        for rerank_model in rerank_models:
            for final_k in final_top_k_values:
                # Determine intermediate K values based on whether reranker is active
                if rerank_model is None:
                    # No reranker: fusion_top_k is the final k
                    current_rerank_top_k = final_k
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
                    HybridStrategy(
                        chunk_strategy_name=chunk_strategy.strategy_name,
                        chunk_size=chunk_strategy.chunk_size,
                        chunk_overlap_ratio=chunk_strategy.chunk_overlap_ratio,
                        summary_model=chunk_strategy.summary_model,
                        summary_prompt_template=chunk_strategy.summary_prompt_template,
                        prompt_target_char_length=chunk_strategy.prompt_target_char_length,
                        summary_truncation_length=chunk_strategy.summary_truncation_length,
                        embedding_model=embed_model,
                        embedding_top_k=current_embedding_top_k,
                        bm25_top_k=current_bm25_top_k,
                        fusion_top_k=current_fusion_top_k,
                        rerank_model=rerank_model,
                        rerank_top_k=current_rerank_top_k
                    )
                )


# --- Combine all strategies to be tested ---
ALL_RETRIEVAL_STRATEGIES: List[Union[BaselineStrategy, HybridStrategy]] = []
ALL_RETRIEVAL_STRATEGIES.extend(BASELINE_STRATEGIES)
# ALL_RETRIEVAL_STRATEGIES.extend(HYPA_STRATEGIES)

print(f"Defined {len(BASELINE_STRATEGIES)} Baseline strategies.")
# print(f"Defined {len(HYPA_STRATEGIES)} HyPA strategies.")
print(f"Total strategies to run: {len(ALL_RETRIEVAL_STRATEGIES)}")
