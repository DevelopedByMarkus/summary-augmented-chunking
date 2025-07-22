import json
from datasets import load_dataset

# Add this to increase timeout globally (optional)
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # Increase default timeout from 10s to 60s

# === CONFIGURATION ===
NUM_CORPUS_DOCS = None     # >0 to download, 0 to skip, None for full
NUM_QA_TASKS = 0         # >0 to download, 0 to skip, None for full

# === CORPUS DOWNLOAD ===
if NUM_CORPUS_DOCS != 0:
    corpus_filename = (
        f"alrag_corpus_first{NUM_CORPUS_DOCS}.jsonl"
        if NUM_CORPUS_DOCS is not None else "alrag_corpus_full.jsonl"
    )
    print(f"Downloading: {NUM_CORPUS_DOCS or 'all'} corpus docs")
    corpus_stream = load_dataset(
        "isaacus/open-australian-legal-corpus",
        split="corpus",
        streaming=True
    )
    with open(corpus_filename, "w", encoding="utf-8") as f:
        for i, example in enumerate(corpus_stream):
            if NUM_CORPUS_DOCS is not None and i >= NUM_CORPUS_DOCS:
                break
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved corpus: {NUM_CORPUS_DOCS or 'all'} docs → {corpus_filename}")
else:
    print("Skipping corpus download (NUM_CORPUS_DOCS is 0)")

# === QA DOWNLOAD ===
if NUM_QA_TASKS != 0:
    qa_filename = (
        f"alrag_qa_first{NUM_QA_TASKS}.jsonl"
        if NUM_QA_TASKS is not None else "alrag_qa_full.jsonl"
    )
    print(f"Downloading: {NUM_QA_TASKS or 'all'} qa tasks")
    qa_dataset = load_dataset(
        "isaacus/open-australian-legal-qa",
        split="train"
    )
    if NUM_QA_TASKS is not None:
        qa_dataset = qa_dataset.select(range(NUM_QA_TASKS))
    with open(qa_filename, "w", encoding="utf-8") as f:
        for entry in qa_dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"Saved QA tasks: {NUM_QA_TASKS or 'all'} entries → {qa_filename}")
else:
    print("Skipping QA task download (NUM_QA_TASKS is 0)")
