import os
from legalbenchrag.methods.baseline import BaselineRetrievalMethod, ChunkingStrategy, \
    RetrievalStrategy as BaselineStrategyType
from legalbenchrag.methods.hypa import HypaRetrievalMethod, HypaStrategy
from legalbenchrag.utils.ai import AIEmbeddingModel, AIRerankModel, AIModel

from legalbenchrag.methods.retrieval_strategies import (
    DEFAULT_SUMMARIZATION_MODEL,
    DEFAULT_SUMMARY_PROMPT_TEMPLATE,
    DEFAULT_PROMPT_TARGET_CHAR_LENGTH,
    DEFAULT_SUMMARY_TRUNCATION_LENGTH
)

PROMPT_TEMPLATE_WITH_CONTEXT = """You are a legal expert. Please answer the following query considering the provided context information.

[Relevant Context Snippets Start]
{formatted_contexts}
[Relevant Context Snippets End]

[Original Query Start]
{original_query_from_base_template}
[Original Query End]

Answer:
"""


# Define your retrieval strategy configurations here
# This dictionary maps a string name (from args.retrieval_strategy) to a full Pydantic model config
RETRIEVAL_STRATEGY_CONFIGS = {
    "s-rcts_oai3S_X": BaselineStrategyType(
        chunking_strategy=ChunkingStrategy(
            strategy_name="summary_rcts",
            chunk_size=500,
            chunk_overlap_ratio=0.0,
            summary_model=DEFAULT_SUMMARIZATION_MODEL,
            summary_prompt_template=DEFAULT_SUMMARY_PROMPT_TEMPLATE,
            prompt_target_char_length=DEFAULT_PROMPT_TARGET_CHAR_LENGTH,
            summary_truncation_length=DEFAULT_SUMMARY_TRUNCATION_LENGTH
        ),
        embedding_model=AIEmbeddingModel(company="openai", model="text-embedding-3-small"),
        embedding_top_k=64,  # Number of candidates to fetch before reranking (or final if no reranker)
        rerank_model=None,  # 'X' implies no reranker
        rerank_top_k=0,  # This value is used by legalbenchrag, set to 0 or an appropriate int
        token_limit=None  # No specific token limit on retrieved content length for now
    ),
    # Example for a non-summary baseline strategy (if you define one later)
    "rcts_oai3S_X": BaselineStrategyType(
        chunking_strategy=ChunkingStrategy(
            strategy_name="rcts",
            chunk_size=500,
            chunk_overlap_ratio=0.0,
        ),
        embedding_model=AIEmbeddingModel(company="openai", model="text-embedding-3-small"),
        embedding_top_k=64,
        rerank_model=None,
        rerank_top_k=0,
        token_limit=None
    ),
    # Add more named strategies here
    # "hypa_default_config": HypaStrategyType(...),
}


def create_retriever(retrieval_strategy_name: str):
    """
    Factory function to create a retrieval method instance based on the strategy name.
    """
    if retrieval_strategy_name not in RETRIEVAL_STRATEGY_CONFIGS:
        raise ValueError(f"Unknown retrieval strategy name: {retrieval_strategy_name}. "
                         f"Available strategies: {list(RETRIEVAL_STRATEGY_CONFIGS.keys())}")

    strategy_config = RETRIEVAL_STRATEGY_CONFIGS[retrieval_strategy_name]

    # Determine which RetrievalMethod class to use based on the type of strategy_config
    # This is a simple check; you might have a more explicit way if strategy names are more structured
    if isinstance(strategy_config, BaselineStrategyType):
        print(f"Creating BaselineRetrievalMethod for strategy: {retrieval_strategy_name}")
        return BaselineRetrievalMethod(retrieval_strategy=strategy_config)
    # elif isinstance(strategy_config, HypaStrategyType):
    #     print(f"Creating HypaRetrievalMethod for strategy: {retrieval_strategy_name}")
    #     return HypaRetrievalMethod(strategy=strategy_config)
    else:
        raise ValueError(f"No matching RetrievalMethod class for strategy config type: {type(strategy_config)}")


def load_corpus_for_dataset(dataset_id: str, corpus_base_path: str = "./data/corpus") -> list:
    """
    Loads all documents from the specified dataset's subdirectory within the corpus.
    Returns a list of legalbenchrag.benchmark_types.Document objects.
    """
    from legalbenchrag.benchmark_types import Document  # Local import to avoid circular dependency if this file grows

    dataset_corpus_path = os.path.join(corpus_base_path, dataset_id)
    corpus_docs = []
    if not os.path.isdir(dataset_corpus_path):
        print(f"Warning: Corpus directory not found for dataset '{dataset_id}' at '{dataset_corpus_path}'.")
        return []

    print(f"Loading corpus documents from: {dataset_corpus_path}")
    for filename in os.listdir(dataset_corpus_path):
        # Assuming documents are text files, adjust if other extensions are used (e.g., .json, .md)
        if filename.endswith((".txt", ".md", ".json")):  # Add more extensions if needed
            file_path_in_corpus = os.path.join(dataset_id, filename)  # Relative path for Document object
            full_file_path = os.path.join(dataset_corpus_path, filename)
            try:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():  # Ensure content is not just whitespace
                    corpus_docs.append(Document(file_path=file_path_in_corpus, content=content))
                else:
                    print(f"Warning: Document '{full_file_path}' is empty. Skipping.")
            except Exception as e:
                print(f"Error reading document '{full_file_path}': {e}. Skipping.")
    print(f"Loaded {len(corpus_docs)} documents for dataset '{dataset_id}'.")
    return corpus_docs


# This function will now replace your existing generate_prompts if it was simpler,
# or you can integrate this logic into your existing generate_prompts.
def generate_prompts_with_rag_context(
        base_prompt_template_text: str,  # This is the content of base_prompt.txt or claude_prompt.txt
        query_text_from_dataset: str,  # This is data_df['text'] or similar
        retrieved_context_strings: list[str]  # List of strings, each a retrieved snippet's content
) -> str:
    """
    Generates a final prompt string by incorporating retrieved contexts into a template
    that wraps the original query derived from base_prompt_template_text and query_text_from_dataset.
    """

    # Step 1: Construct the "original query" part.
    # The base_prompt_template_text usually has a placeholder for the specific query/text from the dataset.
    # Let's assume it's {text} or similar. We need to fill that first.
    # This logic should mirror what your existing generate_prompts(prompt_template, data_df) does
    # for a single item before RAG.
    # For simplicity, assuming base_prompt_template_text might contain "{text}"
    # or is structured such that query_text_from_dataset fits into it.
    # If base_prompt_template_text IS the query structure itself:
    original_query_filled = base_prompt_template_text.replace("{text}", query_text_from_dataset)
    # If base_prompt_template_text is more of a system message and query_text_from_dataset is the actual user query:
    # original_query_filled = f"{base_prompt_template_text}\n\nQuery: {query_text_from_dataset}" (Adjust as needed)

    # Step 2: Format the retrieved contexts.
    if not retrieved_context_strings:
        formatted_contexts = "No relevant context snippets were retrieved."
    else:
        # Enumerate contexts for clarity in the prompt
        formatted_contexts = "\n\n".join(
            [f"Snippet {i + 1}: \n{context}" for i, context in enumerate(retrieved_context_strings)]
        )

    # Step 3: Fill the main RAG prompt template
    final_llm_prompt = PROMPT_TEMPLATE_WITH_CONTEXT.format(
        formatted_contexts=formatted_contexts,
        original_query_from_base_template=original_query_filled
    )

    return final_llm_prompt
