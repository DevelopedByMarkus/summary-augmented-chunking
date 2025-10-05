## Legalbench-RAG Description

LegalBench-RAG is an information retrieval (IR) benchmark, whose purpose is to evaluate any retrieval system against complex legal contract understanding questions. LegalBench-RAG allows the evaluator to deterministically compute precision and recall, even at the exact character level.

## Installation

1. Use the `download_legalbenchrag.py` in the `setup` directory to download the corpus (which serves as the knowledge base) and the benchmark.
2. The corpus needs to be placed in `data/corpus/` and the `data/benchmarks/` relative to the project main directory.
3. Run the `run_benchmark.py` script in the `benchmarks/legalbenchrag/` directory to execute the benchmark.

## Results

1. In case of summary chunking methods, the summary will be saved to `data/summaries/benchmark_name`.
2. The benchmark results will be saved to json files in the `results/legalbenchrag/runname` directory.