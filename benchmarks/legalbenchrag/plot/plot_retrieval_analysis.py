import os
import sys
import argparse
import numpy as np
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


def plot_strategy_results(strategy_name, strategy_data, output_dir, out_name, title):
    """
    Generates and saves a plot for a single strategy's results (including mean and std dev when multiple seeds).

    Args:
        strategy_name (str): The identifier for the strategy.
        strategy_data (dict): Data structured as {top_k: analysis_results}.
        output_dir (Path): Directory to save the plot PNG file.
        out_name (str): Optional custom filename for the plot.
        title (str): Optional title for the plot.
    """
    if not strategy_data:
        print(f"Warning: No data found for strategy '{strategy_name}'. Skipping plot.")
        return

    # Sort data by top_k
    sorted_k = sorted(strategy_data.keys())
    if not sorted_k:
        print(f"Warning: No valid K values found for strategy '{strategy_name}'. Skipping plot.")
        return

    # --- Gather all dataset names seen across any run and any K ---
    all_datasets = set()
    for k in sorted_k:
        for res in strategy_data[k]:
            if res and 'datasets' in res:
                all_datasets.update(res['datasets'].keys())

    # --- Prepare containers for means/stds ---
    overall_mean = []
    overall_std = []

    ds_mean = {ds: [] for ds in all_datasets}
    ds_std = {ds: [] for ds in all_datasets}

    # Helper to compute pct safely
    def pct_overall(res):
        return calculate_wrong_path_percentage(res.get('overall', {})) if res else float('nan')

    def pct_ds(res, ds):
        if not res or 'datasets' not in res or ds not in res['datasets']:
            return float('nan')
        return calculate_wrong_path_percentage(res['datasets'][ds])

    # --- Compute mean/std per K ---
    for k in sorted_k:
        run_results = strategy_data[k]  # list over seeds

        # Overall
        vals = [pct_overall(r) for r in run_results]
        vals_arr = np.array(vals, dtype=float)
        vals_arr = vals_arr[~np.isnan(vals_arr)]
        if vals_arr.size:
            overall_mean.append(np.mean(vals_arr))
            overall_std.append(np.std(vals_arr, ddof=1) if vals_arr.size > 1 else 0.0)
        else:
            overall_mean.append(float('nan'))
            overall_std.append(float('nan'))

        # Each dataset
        for ds in all_datasets:
            dvals = [pct_ds(r, ds) for r in run_results]
            darr = np.array(dvals, dtype=float)
            darr = darr[~np.isnan(darr)]
            if darr.size:
                ds_mean[ds].append(np.mean(darr))
                ds_std[ds].append(np.std(darr, ddof=1) if darr.size > 1 else 0.0)
            else:
                ds_mean[ds].append(float('nan'))
                ds_std[ds].append(float('nan'))

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_xscale("log", base=2)

    x_values = sorted_k

    # Overall mean line + std band
    ax.plot(x_values, overall_mean, label="Weighted Average (Overall)",
            linewidth=2, linestyle='--', color='black', marker='o')

    # Shaded ±1σ band (where both mean and std are finite)
    overall_mean_arr = np.array(overall_mean, dtype=float)
    overall_std_arr = np.array(overall_std, dtype=float)
    valid = np.isfinite(overall_mean_arr) & np.isfinite(overall_std_arr)
    if np.any(valid):
        ax.fill_between(
            np.array(x_values, dtype=float)[valid],
            (overall_mean_arr - overall_std_arr)[valid],
            (overall_mean_arr + overall_std_arr)[valid],
            alpha=0.2, linewidth=0, color='black'
        )

    # Each dataset: mean line + band
    for ds in sorted(all_datasets):
        m = np.array(ds_mean[ds], dtype=float)
        s = np.array(ds_std[ds], dtype=float)
        valid = np.isfinite(m) & np.isfinite(s)
        if not np.any(valid):
            continue
        ax.plot(x_values, m, label=ds, marker='.')
        ax.fill_between(np.array(x_values, dtype=float)[valid],
                        (m - s)[valid], (m + s)[valid],
                        alpha=0.15, linewidth=0)

    if title is not None:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel("Top-K", fontsize=20)
    ax.set_ylabel("IDE(%)", fontsize=20)
    ax.set_ylim(0, 105)  # Y axis from 0 to 100 (with padding)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(fontsize=18)

    # Force ticks at your actual x_values
    ax.set_xticks(x_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # show plain numbers (1, 2, 4, ...)
    ax.ticklabel_format(style='plain', axis='x')  # avoid scientific notation

    ax.tick_params(axis='both', which='major', labelsize=18)  # makes the main numbers bigger

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save plot
    # No need to sanitize filename as user confirmed it's safe
    if out_name is None:
        plot_filename = os.path.join(output_dir, f"{strategy_name}.png")
    else:
        plot_filename = os.path.join(output_dir, out_name)

    try:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot '{plot_filename}': {e}")

    plt.close(fig)  # Close figure to free memory


def main():
    parser = argparse.ArgumentParser(description="Generate plots for retrieval file path analysis based on benchmark results.")
    parser.add_argument("--run_subdirs", nargs="+", type=str, required=True,
                        help="One or more subdirs inside 'results/legalbenchrag' for different seeds.")
    parser.add_argument("--out_dir", type=str, default="plots/legalbenchrag/retrieval_analysis",
                        help="Directory to save the generated plots. This is relative to the current working dir.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Filename of the resulting plot. This must include the file extension (e.g., .png, .pdf).")
    parser.add_argument("--title", type=str, default=None,
                        help="Title for the plot. If not provided, no title will be set.")
    args = parser.parse_args()

    if args.out_name and not args.out_name.endswith(('.png', '.pdf')):
        raise RuntimeError("Error: The output filename must end with .png or .pdf extension, but was:" + args.out_name)

    # --- Determine Paths ---
    project_root = Path.cwd()
    results_base_dir = project_root / "results" / "legalbenchrag"

    # Validate and gather all target dirs
    target_run_dirs = []
    for subdir in args.run_subdirs:
        d = results_base_dir / subdir
        if not os.path.isdir(d):
            print(f"Error: Target directory not found: {d}")
            sys.exit(1)
        target_run_dirs.append(d)

    # --- Scan directory and identify strategies ---
    strategy_files = defaultdict(list)  # {strategy_name: [list of full file paths]}
    print(f"Scanning directories: ")
    for d in target_run_dirs:
        print(f" - {d}")
        for filename in os.listdir(d):
            full_path = os.path.join(d, filename)
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

        # Map each Top-K -> list of analysis_result dicts (one per run that has that K)
        strategy_data = defaultdict(list)  # {top_k: [analysis_results_from_each_run]}

        file_paths = strategy_files[strategy_name]

        # To avoid duplicates within the same run dir, track seen (run_dir, top_k)
        seen_per_dir_k = set()

        for file_path in file_paths:
            filename = os.path.basename(file_path)
            parse_result = parse_filename(filename)
            if not parse_result:
                continue

            _index, _s_name, top_k = parse_result
            run_dir = Path(file_path).parent  # which seed/run this file belongs to

            key = (str(run_dir), top_k)
            if key in seen_per_dir_k:
                print(
                    f"  Warning: Duplicate Top-K {top_k} in {run_dir.name} for '{strategy_name}'. Skipping duplicate {filename}.")
                continue
            seen_per_dir_k.add(key)

            analysis_result = analyze_filepaths_for_plotting(file_path)
            if analysis_result:
                strategy_data[top_k].append(analysis_result)
            else:
                print(f"  Warning: Analysis failed or returned no data for K={top_k} (File: {filename}).")

        # Plot (mean ± std) across runs
        plot_strategy_results(strategy_name, strategy_data, args.out_dir, args.out_name, args.title)

    print("\nPlotting script finished.")


if __name__ == "__main__":
    main()
