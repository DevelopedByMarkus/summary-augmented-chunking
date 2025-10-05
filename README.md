# Summary Augmented Chunking for RAG (SAC-RAG)

**Summary Augmented Chunking for RAG (SAC-RAG)**, is a project designed to improve the reliability of Retrieval-Augmented Generation (RAG) systems, particularly in the legal domain.

Standard RAG pipelines often suffer from a critical failure mode called **Document-Level Retrieval Mismatch (DRM)**, where the retrieval component selects context from entirely incorrect source documents. This project developed a DRM metric for the legal domain, which measures how many times a text snippet is retrieved from an incorrect source document (percentage).

The core contribution is **SAC**, a simple yet effective chunking method. Before chunking, SAC generates a summary of the parent document and prepends it to each chunk. This technique enriches each text chunk with essential document-level context, making the retrieval process more accurate.

Our evaluations show that SAC significantly reduces the DRM and boosts retrieval precision and recall compared to baseline methods across several benchmarks.

## Installation

To use the framework, you need to install the required dependencies.

1. Create an anaconda environment (recommended):
   ```bash
   conda create -n sac_rag python=3.10
   conda activate sac_rag
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the `setup.py` script to setup the `sac_rag` package.
4. Setup the credentials for the AI models you want to use. Copy the `credentials/credentials.example.toml` file to `credentials/credentials.toml` and fill in your API keys for the AI services you plan to use.

## Codebase Overview

The project is organized into two main parts: a standalone retrieval library located in `src/sac_rag` and a set of evaluation benchmarks in the `benchmarks/` directory that use this library to produce the experimental results.

### Core Library: `src/sac_rag`

This directory contains the core, reusable RAG retrieval system. It is designed to be a standalone library that can be integrated into other projects.

*   `utils/retriever_factory.py`: This is the primary entry point for using the library. The `create_retriever` function takes a JSON configuration file and constructs the appropriate retrieval pipeline (currently either `Baseline` or `Hybrid`).
*   `utils/chunking.py`: This module is the heart of the SAC methodology. The `get_chunks` function implements the various chunking strategies. For `summary_naive` and `summary_rcts` strategies, it calls the summarization logic before prepending the summary to each chunk.
*   `methods/`: This package contains the different retrieval implementations.
    *   `baseline.py`: Implements a standard vector-search RAG pipeline.
    *   `hybrid.py`: Implements a hybrid retrieval approach combining sparse (BM25) and dense (vector) search.
*   `utils/ai.py`: This utility module manages all interactions with external or local AI models. It handles API calls or local production for embeddings, reranking, and, crucially, the `generate_document_summary` function which produces the summaries used in the SAC method. It also includes caching into a sqlite database (`data/cache/ai_cache.db/cache.db`) for all AI calls to improve performance.
*   `data_models.py`: Defines the core Pydantic data structures used throughout the library, such as `Document`, `Snippet`, `RetrievedSnippet`, and `QueryResponse`.
*   `utils/config_loader.py`: A helper to load and validate strategy configurations from JSON files into Pydantic models.

### Benchmarks: `benchmarks/`

This directory contains the scripts and code necessary to reproduce the results from our paper. Each subdirectory represents a distinct benchmark that uses the `sac_rag` library.

*   `legalbenchrag/`: This benchmark is focused on evaluating the **retrieval component** of the RAG system. It is build on the work of [Pipitone and Alami (2024)](https://github.com/zeroentropy-ai/legalbenchrag.git).
    *   `run_benchmark.py`: The main script to execute the retrieval tests and calculate metrics like precision, recall, and F1-score at the character overlap level.
    *   `plot/`: Contains scripts to analyze and visualize the results. `analyze_retrieval.py` calculates the Document-Level Retrieval Mismatch (DRM), while `plot_results.py` and `plot_retrieval_analysis.py` generate the performance graphs shown in the paper.
*   `alrag/`: Our custom-built benchmark for **end-to-end evaluation** of legal RAG systems. This is still in progress!
    *   `run_benchmark.py`: This script runs the full pipeline, from question to final answer generation, and evaluates the quality of the generated text against a ground truth answer as well as retrieval precision, recall, F1-Score and DRM.
*   `legalbench/`: Scripts to run experiments on tasks from the established LegalBench suite. It is build on the work of [Guha et al. (2023)](https://github.com/HazyResearch/legalbench.git).
    *   `run_benchmark.py`: The entry point for running the LegalBench tasks.

Each benchmark directory contains a `README.md` file with detailed instructions on how to run the benchmarks, including any required configurations and expected outputs.

## Citations

If you would like to use this work, please cite us!

@article{reuter2025towards,
  title={Towards Reliable Retrieval in RAG Systems for Large Legal Datasets},
  author={Reuter, Markus and Lingenberg Tobias and ...},
  journal={arXiv preprint arXiv:...},
  year={2025},
  url={https://arxiv.org/abs/...}
}

Additionally, here are citations for the datasets we use in this work:

@article{koreeda2021contractnli,
  title={ContractNLI: A dataset for document-level natural language inference for contracts},
  author={Koreeda, Yuta and Manning, Christopher D},
  journal={arXiv preprint arXiv:2110.01799},
  year={2021}
}
@article{hendrycks2021cuad,
  title={Cuad: An expert-annotated nlp dataset for legal contract review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
@article{wang2023maud,
  title={MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding},
  author={Wang, Steven H and Scardigli, Antoine and Tang, Leonard and Chen, Wei and Levkin, Dimitry and Chen, Anya and Ball, Spencer and Woodside, Thomas and Zhang, Oliver and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2301.00876},
  year={2023}
}
@inproceedings{ravichander-etal-2019-question,
    title = "Question Answering for Privacy Policies: Combining Computational and Legal Perspectives",
    author = "Ravichander, Abhilasha  and
      Black, Alan W  and
      Wilson, Shomir  and
      Norton, Thomas  and
      Sadeh, Norman",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1500",
    doi = "10.18653/v1/D19-1500",
    pages = "4949--4959",
    abstract = "Privacy policies are long and complex documents that are difficult for users to read and understand. Yet, they have legal effects on how user data can be collected, managed and used. Ideally, we would like to empower users to inform themselves about the issues that matter to them, and enable them to selectively explore these issues. We present PrivacyQA, a corpus consisting of 1750 questions about the privacy policies of mobile applications, and over 3500 expert annotations of relevant answers. We observe that a strong neural baseline underperforms human performance by almost 0.3 F1 on PrivacyQA, suggesting considerable room for improvement for future systems. Further, we use this dataset to categorically identify challenges to question answerability, with domain-general implications for any question answering system. The PrivacyQA corpus offers a challenging corpus for question answering, with genuine real world utility.",
}
