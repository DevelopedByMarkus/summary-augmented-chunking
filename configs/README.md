This explains the config file arguments

### Embedding Models (used by both)
- oai_embed_model = AIEmbeddingModel(company="openai", model="text-embedding-3-large")
- hf_embed_model_bge_base = AIEmbeddingModel(company="huggingface", model="BAAI/bge-base-en-v1.5")
- hf_embed_model_bge_large = AIEmbeddingModel(company="huggingface", model="BAAI/bge-large-en-v1.5")
- hf_embed_model_gte = AIEmbeddingModel(company="huggingface", model="thenlper/gte-large")
- hf_embed_model_legalbert_base = AIEmbeddingModel(company="huggingface", model="nlpaueb/legal-bert-base-uncased")
- hf_embed_model_legalbert_small = AIEmbeddingModel(company="huggingface", model="nlpaueb/legal-bert-small-uncased")

### Rerank Models (used by both)
- cohere_rerank_model = AIRerankModel(company="cohere", model="rerank-english-v3.0")
- voyage_rerank_model = AIRerankModel(company="voyageai", model="rerank-2-lite")
- hf_rerank_minilm = AIRerankModel(company="huggingface", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
- hf_rerank_bge_base = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-base")
- hf_rerank_bge_large = AIRerankModel(company="huggingface", model="BAAI/bge-reranker-large")

### Define final top_k values to test
- final_top_k_values: list[int] = [1, 2, 4, 8, 16, 32, 64]

### Calculate max K needed for consistent input to reranker caching
- max_final_k = max(final_top_k_values) if final_top_k_values else 64  # Default if list is empty