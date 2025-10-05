## ALRAG Description

ALRAG is our custom-built benchmark for **end-to-end evaluation** of legal RAG systems.

## Installation

1. Execute the `download_dataset.py` in the `setup` directory to download the corpus (which serves as the knowledge base) and the qa-dataset. It is possible to download a subset of the corpus and the qa-dataset, e.g. for debugging purposes. These corpus need to be placed in `data/corpus/alrag/` relative to the project main directory.
2. To be able to run the reduced benchmark version where the corpus consists only of the relevant documents to the qa-paris, execute the `extract_full_qa_documents.py` in the same directory. The qa-files need to be placed in the `data/benchmarks/` directory.
3. Run the `run_benchmark.py` script in the `benchmarks/alrag/` directory to execute the benchmark.

## Evaluation

### Metrics

TODO... This part is still in progress and will be updated later.

### Analysis

Some analysis scripts are provided in the `analysis/` directory to analyze the corpus and qa-dataset.

## Results

1. In case of summary chunking methods, the summary will be saved to `data/summaries/benchmark_name`.
2. The benchmark results will be saved to json files in the `results/alrag/runname` directory.