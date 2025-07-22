import json

# === CONFIGURATION ===
alrag_filename = "alrag_qa_full.jsonl"  # your ALRAG QA dataset
corpus_filename = "alrag_corpus_full.jsonl"
output_filename = "alrag_corpus_gt_only.jsonl"

# === STEP 1: Get used version_ids from ALRAG
used_version_ids = set()

with open(alrag_filename, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        source = entry.get("source", {})
        version_id = source.get("version_id")
        if version_id:
            used_version_ids.add(version_id)

print(f"Found {len(used_version_ids)} unique version_id(s) in ALRAG dataset.")

# === STEP 2: Filter the corpus
matched_docs = 0

with open(corpus_filename, "r", encoding="utf-8") as infile, \
        open(output_filename, "w", encoding="utf-8") as outfile:
    for line in infile:
        entry = json.loads(line)
        if entry.get("version_id") in used_version_ids:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")
            matched_docs += 1

print(f"Written {matched_docs} matched documents to '{output_filename}'")
