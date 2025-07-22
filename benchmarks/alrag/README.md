## ALRAG Description

ALRAG is our custom-built benchmark for **end-to-end evaluation** of legal RAG systems.
- `run_benchmark.py`: This script runs the full pipeline, from question to final answer generation, and evaluates the quality of the generated text against a ground truth answer as well as retrieval precision, recall, F1-Score and PFD.

## Installation

Place the Open Australian Legal Corpus jsonl file in the `data/corpus/alrag` directory. The file should be named `alrag_corpus_full.jsonl`.

TODO...