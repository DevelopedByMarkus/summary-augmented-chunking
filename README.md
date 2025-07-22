# legalbenchrag

`legalbenchrag` is a project for evaluating and improving Retrieval-Augmented Generation (RAG) systems in the legal domain.

The current findings identify a critical failure in standard RAG pipelines—the retrieval of context from entirely incorrect documents—and introduces the **Percentage of Foreign Documents (PFD)** metric to quantify this error.

To address this, the project proposes **Summary Augmented Chunking (SAC)**, a novel method that enriches text chunks with document-level summaries to improve retrieval accuracy.

The project includes three distinct benchmarks for evaluation:
1.  **legalbench-RAG**: A retrieval-focused benchmark to measure the performance of different chunking and retrieval strategies using metrics like precision, recall, and PFD.
2. **legalbench**: An end-to-end evaluation framework for legal tasks, but not tailored for evaluating RAG applications.
3.  **ALRAG**: An end-to-end benchmark to assess the impact of retrieval quality on the final generated answers in a legal question-answering context.

## Installation and Setup

- To use the code it is first necessary to provide the API-keys for the used API models. This can be done by creating a `credentials.toml` file in the `credentials/` directory, following the example provided in `credentials/credentials.example.toml`.
- Secondly, it is best to setup a virtual environment and install the dependencies via the requirements.txt file. Durch development python 3.10 was used.
- `setup.py`: Basic package setup for `legalbenchrag`.
- `data/download_dropbox_data.py`: Script to download the necessary legalbench corpus and benchmark files.
- The corpus must be stored under `data/corpus`, the benchmark files for `legalbenchrag` and `alrag` under `data/benchmarks/`, and the benchmark files for `alqa` under `data/benchmarks/alqa`.
- The summaries and embeddings will be cached in a sqlite database to save time and cost in subsequent runs.

## Codebase Overview

This overview provides a top-down look at the project structure, focusing on the main benchmarking pipelines.

### `legalbenchrag/` — The Core Retrieval Benchmark (legalbench-RAG)

This is the main package for the retrieval-focused benchmark. It evaluates how well different strategies retrieve the correct text snippets.

-   `benchmark.py`: The **primary entry point** for running the retrieval benchmark. It loads the corpus and benchmark data, iterates through all defined retrieval strategies, and executes the evaluation.
-   `run_benchmark.py`: Contains the core evaluation logic. The `run_benchmark` function takes a set of queries, a corpus, and a retrieval method, then calculates precision, recall, and F1-score for the retrieved snippets.
-   `benchmark_types.py`: Defines the essential Pydantic data models used throughout the project, such as `Document`, `Snippet`, `QAGroundTruth`, and the `RetrievalMethod` abstract base class.

-   **`methods/`**: Implements the different retrieval strategies.
    -   `baseline.py`: Implements the `BaselineRetrievalMethod`, which uses a vector database (`sqlite-vec`) for retrieval. It supports various chunking strategies, embedding models, and optional reranking.
    -   `hypa.py`: Implements `HypaRetrievalMethod`, a hybrid approach combining dense vector search and sparse BM25 retrieval, followed by fusion and optional reranking.
    -   `retrieval_strategies.py`: Defines the configurations for all experimental setups. This is where you can define new combinations of chunkers, embedders, and rerankers to test.

-   **`utils/`**: Core utilities supporting the benchmark.
    -   `ai.py`: A crucial module that provides a unified, cached interface for interacting with various AI models (OpenAI, Cohere, Hugging Face) for embedding, reranking, and summarization.
    -   `chunking.py`: Implements the different chunking strategies, including standard `rcts` (Recursive Character Text Splitter) and the novel `summary_rcts` (SAC).

-   **`generate/`**: Contains scripts to process raw datasets (like CUAD, MAUD) and generate the standardized benchmark JSON files.
-   **`plots/`**: Contains scripts to analyze and visualize the benchmark results.

### `alrag/` — The End-to-End QA Benchmark (ALRAG)

This directory contains the implementation of the **ALRAG** benchmark, which evaluates the entire RAG pipeline from question to final answer.

-   `run_benchmark.py`: The **main entry point** for the ALRAG benchmark. It orchestrates the entire process:
    1.  Loads the ALRAG corpus and test set.
    2.  Initializes a retriever (using methods from `legalbenchrag/methods/`) and a generator LLM.
    3.  For each question, it retrieves context, constructs a prompt, and generates an answer.
    4.  Evaluates the generated answer's quality (via semantic similarity) and the retrieval performance.
    5.  Saves detailed results to a CSV file.
-   **`src/`**: Source code specific to the ALRAG benchmark.
    -   `data.py`: Functions to load the Australian Legal QA corpus and test sets.
    -   `evaluation.py`: Implements the evaluation logic, including `evaluate_answer_similarity` using embedding models and document-level retrieval metrics.
    -   `prompts.py`: Defines the prompt templates for the generator LLM, with and without RAG context.
    -   `result_models.py`: Pydantic models for ALRAG test items and result rows.

### `legalbench/` — End-to-End Evaluation on LegalBench Tasks

This directory adapts the original LegalBench framework to run with our custom RAG pipeline, allowing for end-to-end evaluation on a wider range of legal tasks.

-   `run_benchmark.py`: The main script to run LegalBench tasks with RAG. It loads tasks, injects retrieved context into the prompts, generates responses, and evaluates them using the task-specific metrics from LegalBench.
-   **`src/`**: Core components for the RAG-enabled LegalBench runner.
    -   `retrieval.py`: A factory `create_retriever` that uses the strategies defined in `legalbenchrag/methods/`. It defines how retrieved context is formatted and injected into the task prompts.
    -   `generate.py`: A factory `create_generator` to instantiate LLM clients (OpenAI, Cohere, local Hugging Face models) for generating answers.
    -   `evaluation.py`: The standard LegalBench evaluation functions (e.g., `evaluate_exact_match_balanced_accuracy`).


## TODO:

(TODO: the repo has to be renamed and refactored because legalbenchrag is now only a submodule)