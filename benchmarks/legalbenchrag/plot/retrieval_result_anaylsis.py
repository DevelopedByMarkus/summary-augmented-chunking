import json
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_ide_per_document(run_dir: str, filename: str) -> dict:
    """
    Analyzes a result JSON file to count how often snippets from incorrect
    documents were retrieved for each ground truth document.

    This function quantifies the Inter-Document Error (IDE) on a per-file basis.

    Args:
        run_dir: The name of the directory for a specific run, located inside
                 'results/legalbenchrag/'.
        filename: The name of the JSON result file within the run_dir.

    Returns:
        A dictionary where keys are dataset names. Each value is another
        dictionary that maps a ground truth file path to the total count
        of foreign snippets retrieved across all queries associated with that file.
    """
    # Build the path to the results file.
    # The script is expected to be in sac-rag/benchmarks/legalbenchrag/plot/
    # The results are in sac-rag/results/legalbenchrag/
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent.parent
        results_path = project_root / 'results' / 'legalbenchrag' / run_dir / filename
    except NameError:
        # Fallback for interactive environments where __file__ is not defined
        project_root = Path('.').resolve().parent.parent
        results_path = project_root / 'results' / 'legalbenchrag' / run_dir / filename


    if not results_path.is_file():
        raise FileNotFoundError(
            f"Result file not found at the expected path: {results_path}"
        )

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize dictionaries to hold the counts for each dataset
    analysis_results = {
        "contractnli": defaultdict(int),
        "privacy_qa": defaultdict(int),
        "cuad": defaultdict(int),
        "maud": defaultdict(int),
    }

    # Iterate over each question-answering item in the results
    for item in data.get("qa_result_list", []):
        qa_gt = item.get("qa_gt", {})
        retrieved_snippets = item.get("retrieved_snippets", [])

        if not qa_gt or not retrieved_snippets:
            continue

        # Identify the dataset from the tags
        dataset = qa_gt.get("tags", [None])[0]
        if dataset not in analysis_results:
            continue

        # Create a set of all correct source document file paths for this query
        gt_filepaths = {snippet["file_path"] for snippet in qa_gt.get("snippets", [])}
        if not gt_filepaths:
            continue

        # Count how many retrieved snippets are from a "wrong" document
        foreign_snippet_count = 0
        for retrieved in retrieved_snippets:
            if retrieved.get("file_path") not in gt_filepaths:
                foreign_snippet_count += 1

        # If any foreign snippets were found, add this count to the total
        # for each of the ground truth documents involved in this query.
        if foreign_snippet_count > 0:
            for gt_path in gt_filepaths:
                analysis_results[dataset][gt_path] += foreign_snippet_count

    # Convert defaultdicts to regular dicts for the final output
    final_results = {
        dataset: dict(counts) for dataset, counts in analysis_results.items()
    }
    return final_results


if __name__ == '__main__':
    # This executable block allows you to run the script directly from the command line.
    parser = argparse.ArgumentParser(
        description="Analyze Inter-Document Errors (IDE) from a SAC-RAG result file."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="base_s-rcts-500-X_e-oai3L-r-X",
        help="The specific run directory containing the result file."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="0_base_rcts-500-X__e-oai3L_r-X_1.json",
        help="The name of the result JSON file to analyze."
    )
    args = parser.parse_args()

    try:
        ide_analysis = analyze_ide_per_document(run_dir=args.run_dir, filename=args.filename)

        print(f"Analyzing results from: {args.run_dir}/{args.filename}")
        print("\n--- Inter-Document Error (IDE) Analysis per Document ---\n")
        for dataset, file_counts in ide_analysis.items():
            print(f"--- Dataset: {dataset} ---")
            if not file_counts:
                print("No foreign documents were retrieved for any query in this dataset.")
            else:
                # Sort by filename for consistent and readable output
                sorted_items = sorted(file_counts.items())
                print(f"{'Ground Truth File': <50} | {'Count of Foreign Snippets'}")
                print(f"{'-'*50}-|----------------------------")
                for filename, count in sorted_items:
                    print(f"{filename:<50} | {count}")
            print("-" * (len(dataset) + 18) + "\n")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the script is run from the 'sac-rag/benchmarks/legalbenchrag/plot/' directory or that the project structure is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
