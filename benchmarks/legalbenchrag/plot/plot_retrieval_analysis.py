import os
import sys
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import the analysis function ---
# Assuming analyze_retrieval.py is in the same directory
try:
    from .analyze_retrieval import analyze_filepaths_for_plotting
except ImportError:
    print("Error: Could not import 'analyze_filepaths_for_plotting' from 'analyze_retrieval.py'.")
    print("Ensure 'analyze_retrieval.py' is in the same directory as this script.")
    sys.exit(1)


def parse_filename(filename):
    """
    Parses the filename to extract index, strategy name, and top_k.
    Format: {index}_{strategy_name}_{top_k}.json
    Returns (index, strategy_name, top_k) or None if format is invalid.
    """
    if not filename.endswith(".json"):
        return None

    base_name = filename[:-5]  # Remove .json
    parts = base_name.split('_')

    if len(parts) < 3:
        # print(f"Warning: Filename '{filename}' has too few parts separated by '_'. Skipping.")
        return None # Needs at least index, some strategy name, top_k

    index_str = parts[0]
    top_k_str = parts[-1][1:]  # Remove leading 'k' from top_k
    strategy_name = '_'.join(parts[1:-1])

    if not index_str.isdigit() or not top_k_str.isdigit():
        # print(f"Warning: Index ('{index_str}') or TopK ('{top_k_str}') in '{filename}' is not numeric. Skipping.")
        return None

    return int(index_str), strategy_name, int(top_k_str)


def calculate_wrong_path_percentage(stats):
    """Calculates the percentage of incorrect file paths."""
    correct = stats.get('correct', 0)
    incorrect = stats.get('incorrect', 0)
    total = correct + incorrect
    if total == 0:
        return 0.0
    return (incorrect / total) * 100


def plot_strategy_results(strategy_name, strategy_data, output_dir):
    """
    Generates and saves a plot for a single strategy's results.

    Args:
        strategy_name (str): The identifier for the strategy.
        strategy_data (dict): Data structured as {top_k: analysis_results}.
        output_dir (Path): Directory to save the plot PNG file.
    """
    if not strategy_data:
        print(f"Warning: No data found for strategy '{strategy_name}'. Skipping plot.")
        return

    # Sort data by top_k
    sorted_k = sorted(strategy_data.keys())
    if not sorted_k:
        print(f"Warning: No valid K values found for strategy '{strategy_name}'. Skipping plot.")
        return

    # --- Prepare data for plotting ---
    x_values = sorted_k
    y_values_overall = []
    y_values_datasets = defaultdict(list)
    all_datasets = set()
    legend_counts = {}

    # Get dataset names and counts from the first K value's results
    first_k_results = strategy_data[sorted_k[0]]
    if first_k_results and 'datasets' in first_k_results:
        for ds_name, ds_stats in first_k_results['datasets'].items():
            all_datasets.add(ds_name)
            legend_counts[ds_name] = ds_stats.get('total', 0)
    else:
        print(f"Warning: Could not extract dataset info from first K value for strategy '{strategy_name}'. Legend might be incomplete.")

    # Calculate percentages for each K
    for k in sorted_k:
        results = strategy_data[k]
        if not results: # Should not happen if collected correctly, but safety check
            print(f"Warning: Missing analysis results for K={k} in strategy '{strategy_name}'. Plot may be incomplete.")
            # Add NaN or skip? Let's add NaN for plotting continuity break
            y_values_overall.append(float('nan'))
            for ds_name in all_datasets:
                y_values_datasets[ds_name].append(float('nan'))
            continue

        # Overall percentage
        overall_pct = calculate_wrong_path_percentage(results.get('overall', {}))
        y_values_overall.append(overall_pct)

        # Per-dataset percentage
        processed_datasets_for_k = set()
        if 'datasets' in results:
            for ds_name, ds_stats in results['datasets'].items():
                ds_pct = calculate_wrong_path_percentage(ds_stats)
                y_values_datasets[ds_name].append(ds_pct)
                processed_datasets_for_k.add(ds_name)

        # Ensure all datasets have a value for this K (use NaN if missing)
        missing_datasets = all_datasets - processed_datasets_for_k
        for ds_name in missing_datasets:
            y_values_datasets[ds_name].append(float('nan'))

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot overall weighted average
    ax.plot(x_values, y_values_overall, label="Weighted Average (Overall)",
            linewidth=2, linestyle='--', color='black', marker='o')

    # Plot each dataset, sorted alphabetically for consistent legend order
    sorted_legend_datasets = sorted(list(all_datasets))
    for ds_name in sorted_legend_datasets:
        total_count = legend_counts.get(ds_name, 'N/A')
        label = f"{ds_name} ({total_count})"
        if ds_name in y_values_datasets and len(y_values_datasets[ds_name]) == len(x_values):
            ax.plot(x_values, y_values_datasets[ds_name], label=label, marker='.')
        else:
            print(f"Warning: Data length mismatch or missing data for dataset '{ds_name}' in strategy '{strategy_name}'. Skipping its line.")

    # ax.set_title(f"Retrieval Analysis: Wrong File Path %\nStrategy: {strategy_name}", fontsize=14)
    ax.set_xlabel("Top-K", fontsize=17)
    ax.set_ylabel("PFD(%)", fontsize=17)
    ax.set_ylim(0, 105)  # Y axis from 0 to 100 (with padding)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=15)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    # No need to sanitize filename as user confirmed it's safe
    plot_filename = os.path.join(output_dir, f"_{strategy_name}.png")
    # plot_filename = os.path.join(output_dir, "pfd_baseline.pdf")
    try:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot '{plot_filename}': {e}")

    plt.close(fig)  # Close figure to free memory


def main():
    parser = argparse.ArgumentParser(description="Generate plots for retrieval file path analysis based on benchmark results.")
    parser.add_argument("--run_subdir",
                        help="Name of the run subdirectory inside 'benchmark_results'. Example: '2024-01-15_10-30-00'")
    args = parser.parse_args()

    # --- Determine Paths ---
    project_root = Path.cwd()
    results_base_dir = project_root / "results" / "legalbenchrag"
    target_run_dir = results_base_dir / args.run_subdir
    plot_output_dir = project_root / "plots" / "legalbenchrag" / "retrieval_analysis"

    if not os.path.isdir(target_run_dir):
        print(f"Error: Target directory not found: {target_run_dir}")
        sys.exit(1)

    # --- Scan directory and identify strategies ---
    strategy_files = defaultdict(list)  # {strategy_name: [list of full file paths]}
    print(f"Scanning directory: {target_run_dir}")

    for filename in os.listdir(target_run_dir):
        full_path = os.path.join(target_run_dir, filename)
        if os.path.isfile(full_path) and filename.endswith(".json"):
            parse_result = parse_filename(filename)
            if parse_result:
                _index, strategy_name, _top_k = parse_result
                strategy_files[strategy_name].append(full_path)
            else:
                print(f"Warning: Skipping file with unexpected name format: {filename}")

    if not strategy_files:
        print("Error: No valid strategy result files found in the specified directory.")
        sys.exit(1)

    # --- User Selection ---
    strategies = sorted(strategy_files.keys())
    print("\nFound Strategies:")
    for idx, name in enumerate(strategies):
        print(f"  [{idx}] {name}")
    print("  [-1] Plot ALL strategies")

    try:
        user_choice = input("Enter the index of the strategy to plot (or -1 for all): ")
        selected_index = int(user_choice)
    except ValueError:
        print(f"Invalid input '{user_choice}'. Please enter a number.")
        sys.exit(1)

    strategies_to_plot = []
    if selected_index == -1:
        strategies_to_plot = strategies
        print("\nSelected: Plotting ALL strategies.")
    elif 0 <= selected_index < len(strategies):
        strategies_to_plot = [strategies[selected_index]]
        print(f"\nSelected: Plotting strategy '{strategies_to_plot[0]}'.")
    else:
        print(f"Invalid index '{selected_index}'. Valid indices are 0 to {len(strategies)-1} or -1.")
        sys.exit(1)

    # --- Process and Plot Selected Strategies ---
    for strategy_name in strategies_to_plot:
        print(f"\nProcessing strategy: {strategy_name}")
        strategy_data = {} # {top_k: analysis_results}
        processed_k_values = set()

        file_paths = strategy_files[strategy_name]

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            parse_result = parse_filename(filename)
            if not parse_result: continue # Should not happen if collected correctly

            _index, _s_name, top_k = parse_result

            if top_k in processed_k_values:
                print(f"  Warning: Duplicate Top-K value {top_k} found for strategy '{strategy_name}' (File: {filename}). Skipping duplicate.")
                continue

            # Run analysis
            analysis_result = analyze_filepaths_for_plotting(file_path)

            if analysis_result:
                strategy_data[top_k] = analysis_result
                processed_k_values.add(top_k)
                # print(f"  Analyzed K={top_k} from {filename}") # Verbose
            else:
                print(f"  Warning: Analysis failed or returned no data for K={top_k} (File: {filename}).")

        # Plot the collected data for this strategy
        plot_strategy_results(strategy_name, strategy_data, plot_output_dir)

    print("\nPlotting script finished.")


if __name__ == "__main__":
    main()
