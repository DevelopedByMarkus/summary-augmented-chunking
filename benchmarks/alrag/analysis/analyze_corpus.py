import json
import statistics

# corpus_filename = "alrag_corpus_full.jsonl"
corpus_filename = "alrag_corpus_gt_only.jsonl"

# Load the JSONL file
lengths = []
with open(corpus_filename, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        text = entry.get("text", "")
        if text:
            lengths.append(len(text))

# Compute and print statistics
if lengths:
    avg_length = sum(lengths) / len(lengths)
    median_length = statistics.median(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    print(f"\nSource text file (character count):")
    print(f"  Avg:    {avg_length:.2f}")
    print(f"  Median: {median_length:.2f}")
    print(f"  Min:    {min_length}")
    print(f"  Max:    {max_length}")
else:
    print("No valid text data found.")
