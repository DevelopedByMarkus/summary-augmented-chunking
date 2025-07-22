import json
import re

# === CONFIGURATION ===
input_filename = "open_australian_legal_corpus_full.jsonl"

# Provide one of these:
target_version_id = None  # e.g., "legislation/act/1234"
target_citation = "Smith v The State of New South Wales (2015)"  # or None

if not target_version_id and not target_citation:
    raise ValueError("You must provide either target_version_id or target_citation.")
if target_version_id and target_citation:
    raise ValueError("You can only provide one of target_version_id or target_citation.")


# === FUNCTION TO CLEAN FILENAMES ===
def make_safe_filename(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[\\/*?:"<>| \t]', '_', value)
    return value


# === SEARCH AND SAVE ===
found = False

with open(input_filename, "r", encoding="utf-8") as infile:
    for line in infile:
        entry = json.loads(line)

        if target_version_id and entry.get("version_id") == target_version_id:
            match_key = target_version_id
            found = True
        elif target_citation and entry.get("citation") == target_citation:
            match_key = target_citation
            found = True

        if found:
            safe_name = make_safe_filename(match_key)
            output_filename = f"matched_entry_{safe_name}.json"

            with open(output_filename, "w", encoding="utf-8") as outfile:
                json.dump(entry, outfile, ensure_ascii=False, indent=2)

            print(f"Match saved to '{output_filename}'")
            break

if not found:
    print("No matching entry found.")
