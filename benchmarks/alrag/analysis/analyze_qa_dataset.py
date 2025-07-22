import json
import statistics

# === CONFIGURATION ===
qa_filename = "open_australian_legal_qa_full.jsonl"  # Adjust if needed

# === DATA COLLECTION ===
question_lengths = []
answer_lengths = []
snippet_lengths = []

with open(qa_filename, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        source = entry.get("source", {})
        snippet = source.get("text", "")

        question_lengths.append(len(question))
        answer_lengths.append(len(answer))
        snippet_lengths.append(len(snippet))


# === STATISTICS FUNCTION ===
def print_stats(name, values):
    print(f"\n{name} (character count):")
    print(f"  Avg:    {sum(values)/len(values):.2f}")
    print(f"  Median: {statistics.median(values):.2f}")
    print(f"  Min:    {min(values)}")
    print(f"  Max:    {max(values)}")


# === PRINT RESULTS ===
print_stats("Question", question_lengths)
print_stats("Answer", answer_lengths)
print_stats("Source text snippet", snippet_lengths)
