ABBREVIATIONS = {
    "embedding": {
        "text-embedding-3-large": "oai3L",
        "text-embedding-3-small": "oai3S",
        "BAAI/bge-base-en-v1.5": "bgeB",
        "BAAI/bge-large-en-v1.5": "bgeL",
        "thenlper/gte-large": "gteL",
        "nlpaueb/legal-bert-base-uncased": "LbertB",
        "nlpaueb/legal-bert-small-uncased": "LbertS",
        "gpt-4o-mini": "g4omini"
    },
    "reranker": {
        "rerank-english-v3.0": "coh3",
        "rerank-2-lite": "voy2l",
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "miniLM",
        "BAAI/bge-reranker-base": "bgeB",
        "BAAI/bge-reranker-large": "bgeL",
        None: "X"
    },
    "chunking": {
        "rcts": "rcts",
        "naive": "naive",
        "summary_rcts": "s-rcts",
        "summary_naive": "s-naive",
    },
    "method": {
        "baseline": "base",
        "hybrid": "hybrid",
    }
}
