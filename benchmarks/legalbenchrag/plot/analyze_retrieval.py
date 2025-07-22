import json
import os
import argparse
from collections import defaultdict
import sys


def analyze_filepaths_for_plotting(json_file_path):
    """
    Analyzes retrieval file path accuracy and returns structured data for plotting.

    Args:
        json_file_path (str): The full path to the benchmark results JSON file.

    Returns:
        dict | None: A dictionary containing structured statistics suitable for plotting:
            {
              'datasets': {
                  'dataset_name': {'correct': int, 'incorrect': int, 'total': int},
                  ...
              },
              'overall': {'correct': int, 'incorrect': int, 'total': int}
            }
        Returns None if the file cannot be processed or contains no valid data.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:  # Catching generic exception for simplicity here
        # Original function prints specific errors, this one returns None on any read error
        # print(f"Error reading or parsing {json_file_path}: {e}", file=sys.stderr)
        return None

    dataset_stats_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

    if 'qa_result_list' not in data:
        # print(f"Error: 'qa_result_list' key not found in {json_file_path}", file=sys.stderr)
        return None

    valid_item_found = False
    for item in data['qa_result_list']:
        try:
            qa_gt = item.get('qa_gt', {})
            gt_snippets = qa_gt.get('snippets', [])
            tags = qa_gt.get('tags', [])
            retrieved_snippets = item.get('retrieved_snippets', [])

            if not gt_snippets or not tags or not retrieved_snippets:
                continue  # Skip items missing essential parts

            dataset_tag = tags[0]
            ground_truth_filepaths = set(snippet.get('file_path') for snippet in gt_snippets if snippet.get('file_path'))

            if not ground_truth_filepaths:
                continue  # Skip if ground truth file paths are missing

            valid_item_found = True # Mark that we found at least one processable item

            for snippet in retrieved_snippets:
                retrieved_filepath = snippet.get('file_path')
                # Count as incorrect if filepath is missing or doesn't match any ground truth path
                if not retrieved_filepath or retrieved_filepath not in ground_truth_filepaths:
                    dataset_stats_counts[dataset_tag]['incorrect'] += 1
                else:
                    dataset_stats_counts[dataset_tag]['correct'] += 1
        except Exception:
            # Ignore errors within a single item processing for robustness in plotting
            # print(f"Warning: Error processing an item in {json_file_path}. Skipping item.", file=sys.stderr)
            continue

    if not valid_item_found:
        # print(f"Warning: No processable QA items found in {json_file_path}", file=sys.stderr)
        return None # Return None if no data could be extracted

    # --- Post-processing for plotting structure ---
    plot_data = {'datasets': {}, 'overall': {}}
    overall_correct = 0
    overall_incorrect = 0

    # Ensure dataset_stats_counts is not empty before proceeding
    if not dataset_stats_counts:
        # print(f"Warning: No dataset statistics were generated for {json_file_path}", file=sys.stderr)
        return None  # Return None if, despite valid items, no stats were generated (edge case)

    for dataset, counts in dataset_stats_counts.items():
        correct = counts.get('correct', 0)  # Use .get for safety
        incorrect = counts.get('incorrect', 0)
        total = correct + incorrect
        # Only include datasets with actual results
        if total > 0:
            plot_data['datasets'][dataset] = {
                'correct': correct,
                'incorrect': incorrect,
                'total': total
            }
            overall_correct += correct
            overall_incorrect += incorrect

    overall_total = overall_correct + overall_incorrect
    plot_data['overall'] = {
        'correct': overall_correct,
        'incorrect': overall_incorrect,
        'total': overall_total
    }

    # Return None if overall total is 0, meaning no snippets were actually analyzed
    if overall_total == 0:
        # print(f"Warning: Overall total snippets analyzed is 0 for {json_file_path}", file=sys.stderr)
        return None

    return plot_data


def analyze_filepaths(json_file_path):
    """
    Analyzes the retrieval file path accuracy from a legalbench-RAG results JSON file.

    Args:
        json_file_path (str): The full path to the benchmark results JSON file.

    Returns:
        tuple: A tuple containing:
            - dict: Statistics per dataset ({dataset: {'correct': int, 'incorrect': int}}).
            - dict: Overall statistics ({'correct': int, 'incorrect': int}).
        Returns (None, None) if the file cannot be processed.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_file_path}", file=sys.stderr)
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_file_path}: {e}", file=sys.stderr)
        return None, None

    dataset_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
    overall_stats = {'correct': 0, 'incorrect': 0}

    if 'qa_result_list' not in data:
        print(f"Error: 'qa_result_list' key not found in {json_file_path}", file=sys.stderr)
        return None, None

    for item in data['qa_result_list']:
        try:
            # --- Ground Truth Info ---
            qa_gt = item.get('qa_gt', {})
            gt_snippets = qa_gt.get('snippets', [])
            tags = qa_gt.get('tags', [])

            if not gt_snippets:
                print(f"Warning: Skipping item with no ground truth snippets: Query='{qa_gt.get('query', 'N/A')}'", file=sys.stderr)
                continue

            if not tags:
                print(f"Warning: Skipping item with no dataset tags: Query='{qa_gt.get('query', 'N/A')}'", file=sys.stderr)
                continue

            # Assuming one primary dataset tag per item
            dataset_tag = tags[0]

            # Collect all unique ground truth file paths
            ground_truth_filepaths = set(snippet.get('file_path') for snippet in gt_snippets if snippet.get('file_path'))
            if not ground_truth_filepaths:
                 # print(f"Warning: Skipping item with no 'file_path' in ground truth snippets: Query='{qa_gt.get('query', 'N/A')}'", file=sys.stderr)
                 continue # Skip if ground truth file paths are missing

            # --- Retrieved Snippets Info ---
            retrieved_snippets = item.get('retrieved_snippets', [])

            # As per user confirmation, we assume retrieved_snippets is never empty.
            # If it could be, add a check here: if not retrieved_snippets: continue

            for snippet in retrieved_snippets:
                retrieved_filepath = snippet.get('file_path')

                if not retrieved_filepath:
                    # print(f"Warning: Retrieved snippet missing 'file_path': Query='{qa_gt.get('query', 'N/A')}'", file=sys.stderr)
                    # Decide how to handle this: count as incorrect or skip? Let's count as incorrect.
                    dataset_stats[dataset_tag]['incorrect'] += 1
                    overall_stats['incorrect'] += 1
                    continue

                # Check if the retrieved path matches ANY ground truth path
                if retrieved_filepath in ground_truth_filepaths:
                    dataset_stats[dataset_tag]['correct'] += 1
                    overall_stats['correct'] += 1
                else:
                    dataset_stats[dataset_tag]['incorrect'] += 1
                    overall_stats['incorrect'] += 1

        except KeyError as e:
            print(f"Warning: Missing expected key {e} in an item. Skipping.", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Warning: An unexpected error occurred processing an item: {e}. Skipping.", file=sys.stderr)
            continue

    return dataset_stats, overall_stats


def format_results(dataset_stats, overall_stats, input_filename):
    """Formats the analysis results into a string."""
    output_lines = []
    output_lines.append(f"Analysis for: {input_filename}")
    output_lines.append("-" * 30)

    # Sort datasets alphabetically for consistent output
    sorted_datasets = sorted(dataset_stats.keys())

    for dataset in sorted_datasets:
        stats = dataset_stats[dataset]
        correct = stats['correct']
        incorrect = stats['incorrect']
        total = correct + incorrect
        percent_correct = (correct / total * 100) if total > 0 else 0
        percent_incorrect = (incorrect / total * 100) if total > 0 else 0

        output_lines.append(f"\nDataset: {dataset}")
        output_lines.append(f"  Correct File Paths:   {correct:>5} / {total} ({percent_correct:.2f}%)")
        output_lines.append(f"  Incorrect File Paths: {incorrect:>5} / {total} ({percent_incorrect:.2f}%)")
        output_lines.append(f"  Total Retrieved Snippets: {total}")

    output_lines.append("\n" + "-" * 30)
    output_lines.append("--- TOTAL ---")

    total_correct = overall_stats['correct']
    total_incorrect = overall_stats['incorrect']
    grand_total = total_correct + total_incorrect
    grand_percent_correct = (total_correct / grand_total * 100) if grand_total > 0 else 0
    grand_percent_incorrect = (total_incorrect / grand_total * 100) if grand_total > 0 else 0

    output_lines.append(f"  Correct File Paths:   {total_correct:>5} / {grand_total} ({grand_percent_correct:.2f}%)")
    output_lines.append(f"  Incorrect File Paths: {total_incorrect:>5} / {grand_total} ({grand_percent_incorrect:.2f}%)")
    output_lines.append(f"  Total Retrieved Snippets: {grand_total}")
    output_lines.append("-" * 30)

    return "\n".join(output_lines)


def save_results_to_file(results_string, output_filepath):
    """Saves the results string to a text file."""
    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(results_string)
        print(f"\nResults successfully saved to: {output_filepath}")
    except IOError as e:
        print(f"\nError: Could not write results to file {output_filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nAn unexpected error occurred while saving results: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG retrieval file path accuracy from legalbench-RAG JSON results.")
    parser.add_argument("results_path",
                        help="Path to the JSON results file, relative to the 'benchmark_results' directory. "
                             "Example: 'base_hypa_gpt_naive_rcts/2_baseline_text-embedding-3-large.json'")

    args = parser.parse_args()

    # Construct the full path relative to the script's assumed location (project root)
    base_dir = "benchmark_results"
    full_json_path = os.path.join(base_dir, args.results_path)

    print(f"Starting analysis for: {full_json_path}")

    dataset_stats, overall_stats = analyze_filepaths(full_json_path)
    print("dataset #: ", len(dataset_stats))

    if dataset_stats is None or overall_stats is None:
        print("Analysis aborted due to errors.", file=sys.stderr)
        sys.exit(1) # Exit with error code

    if not overall_stats['correct'] and not overall_stats['incorrect']:
         print("Warning: No retrieved snippets were processed. Check the input file format and content.", file=sys.stderr)
         # Decide if exiting or proceeding with empty results is better. Let's proceed.
         # sys.exit(1)

    # Format results for display and saving
    results_output = format_results(dataset_stats, overall_stats, full_json_path)

    # Print results to console
    print("\n" + results_output)

    # Determine output file path
    output_dir = os.path.dirname(full_json_path)
    base_filename = os.path.basename(full_json_path)
    filename_without_ext, _ = os.path.splitext(base_filename)
    output_filename = f"{filename_without_ext}_filepath_analysis.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    # Save results to text file
    save_results_to_file(results_output, output_filepath)

if __name__ == "__main__":
    main()
