import re
from legalbenchrag.methods.baseline import RetrievalStrategy as BaselineStrategy
from legalbenchrag.methods.hypa import HypaStrategy

# --- Abbreviation Mappings ---
ABBREVIATIONS = {
    "embedding": {
        "text-embedding-3-large": "oai3L",
        "BAAI/bge-base-en-v1.5": "bgeB",
        "BAAI/bge-large-en-v1.5": "bgeL",
        "thenlper/gte-large": "gteL",
        "nlpaueb/legal-bert-base-uncased": "LbertB",
        "nlpaueb/legal-bert-small-uncased": "LbertS",
    },
    "reranker": {
        "rerank-english-v3.0": "coh",
        "rerank-2-lite": "voy",
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "miniLM",
        "BAAI/bge-reranker-base": "bgeB",
        "BAAI/bge-reranker-large": "bgeL",
        # Add None mapping for placeholder
        None: "X"
    },
    "chunking": {
        "rcts": "rcts",
        "naive": "naive",
    },
    "method": {
        "baseline": "base",
        "hypa": "hypa",
    }
}

DEFAULT_ABBR = "unk"
NONE_ABBR = "X"  # Placeholder for when a component (like reranker) is not used


def get_abbreviation(value: str | None, category: str) -> str:
    """Gets the abbreviation for a given value and category."""
    if value is None and category == "reranker":
        return NONE_ABBR  # Special handling for optional reranker

    category_map = ABBREVIATIONS.get(category)
    if not category_map:
        print(f"Warning: Unknown abbreviation category '{category}'. Using default '{DEFAULT_ABBR}'.")
        return DEFAULT_ABBR

    abbr = category_map.get(value)  # type: ignore
    if abbr is None:
        # Check again explicitly for None in case it wasn't the reranker category
        if value is None:
            return NONE_ABBR
        # Value is not None, but not found in map
        print(f"Warning: Unknown value '{value}' for category '{category}'. Using default '{DEFAULT_ABBR}'.")
        return DEFAULT_ABBR
    return abbr


def generate_filename(index: int, retrieval_strategy, row: dict) -> str:
    """Generates the detailed filename based on the strategy parameters."""
    try:
        # --- Method ---
        method_name = row.get("method", "unknown")
        method_abbr = get_abbreviation(method_name, "method")

        # --- Chunking ---
        # Access chunking info - needs slight adjustment based on strategy type
        chunk_strategy_name = None
        chunk_size = None
        chunk_overlap = 0  # Default overlap placeholder
        chunk_overlap_str = str(chunk_overlap) if chunk_overlap > 0 else NONE_ABBR

        if isinstance(retrieval_strategy, BaselineStrategy):
            chunk_strategy_name = getattr(retrieval_strategy.chunking_strategy, 'strategy_name', None)
            chunk_size = getattr(retrieval_strategy.chunking_strategy, 'chunk_size', None)

        elif isinstance(retrieval_strategy, HypaStrategy):
            chunk_strategy_name = getattr(retrieval_strategy, 'chunk_strategy_name', None)
            chunk_size = getattr(retrieval_strategy, 'chunk_size', None)

        chunk_abbr = get_abbreviation(chunk_strategy_name, "chunking")
        chunk_size_str = str(chunk_size) if chunk_size is not None else NONE_ABBR

        # --- Embedding ---
        embed_model_name = getattr(getattr(retrieval_strategy, 'embedding_model', None), 'model', None)
        embed_abbr = get_abbreviation(embed_model_name, "embedding")

        # --- Reranking ---
        rerank_model_obj = getattr(retrieval_strategy, 'rerank_model', None)
        rerank_model_name = getattr(rerank_model_obj, 'model', None) if rerank_model_obj else None
        rerank_abbr = get_abbreviation(rerank_model_name, "reranker")  # Handles None case -> 'X'

        # --- Top-K ---
        rerank_k = getattr(retrieval_strategy, 'rerank_top_k', None)
        rerank_k_str = str(rerank_k) if rerank_k is not None else NONE_ABBR

        # --- Construct Filename ---
        # Format: {index}_{method}_{chunkingStrategy}-{chunkSize}-{chunkOverlapRatio}_e-{embedModel}_r-{rerankModel}_{rerankTopK}
        filename = (
            f"{index}_{method_abbr}_{chunk_abbr}-{chunk_size_str}-{chunk_overlap_str}"
            f"_e-{embed_abbr}"
            f"_r-{rerank_abbr}_{rerank_k_str}"
        )
        return filename

    except AttributeError as e:
        print(f"Error generating filename for index {index}: Missing attribute {e}. Using fallback name.")
        # Fallback to a simpler name if attributes are missing
        method_name = row.get("method", "unknown")
        embed_name = str(row.get('embedding_model_name', 'unknown')).replace('/', '_')  # Basic sanitize
        return f"{index}_{method_name}_{embed_name}_ERROR_FALLBACK"
    except Exception as e:
        print(f"Unexpected error generating filename for index {index}: {e}. Using fallback name.")
        method_name = row.get("method", "unknown")
        embed_name = str(row.get('embedding_model_name', 'unknown')).replace('/', '_')
        return f"{index}_{method_name}_{embed_name}_ERROR_FALLBACK"


# Define characters typically illegal in Windows filenames and replacement
ILLEGAL_FILENAME_CHARS = r'[<>:"|?*]'
REPLACEMENT_CHAR = '_'


def sanitize_filename(filename: str) -> str:
    """Replaces characters illegal in Windows filenames with underscores."""
    return re.sub(ILLEGAL_FILENAME_CHARS, REPLACEMENT_CHAR, filename)
