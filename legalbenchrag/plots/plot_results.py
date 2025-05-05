import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.lines import Line2D
import sys
from collections import defaultdict

# --- Constants ---

# Abbreviation dictionary provided by the user
ABBREVIATIONS = {
    "embedding": {
        "text-embedding-3-large": "oai3L",
        "BAAI/bge-base-en-v1.5": "bgeB",
        "BAAI/bge-large-en-v1.5": "bgeL",
        "thenlper/gte-large": "gteL",
        "nlpaueb/legal-bert-base-uncased": "LbertB",
        "nlpaueb/legal-bert-small-uncased": "LbertS",
        "text-embedding-3-small": "oai3S",  # Added for completeness if needed
        "text-embedding-ada-002": "oaiAda",  # Added for completeness if needed
    },
    "reranker": {
        "rerank-english-v3.0": "coh",
        "rerank-2-lite": "voy",  # Changed from 'voy' to 'voyage' for clarity maybe? Assuming 'voy' is intended.
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "miniLM",
        "BAAI/bge-reranker-base": "bgeRB",  # Adjusted for consistency
        "BAAI/bge-reranker-large": "bgeRL",  # Adjusted for consistency
        "<None>": "X"  # Special key for handling None/NaN
    },
    "chunking": {
        "rcts": "rcts",
        "naive": "naive",
    },
    "method": {
        "baseline": "base",
        "hypa": "hypa",
    }
}

IDENTIFYING_COLS = ['method', 'embedding_model_name', 'chunk_strategy_name', 'rerank_model_name']
METRICS_TO_PLOT = {'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1-Score'}
LINESTYLES = ['-', '--', '-.', ':']
DEFAULT_K = 64  # Default K if none found, though the logic should prevent this


# --- Helper Functions ---


def get_k_value(row):
    """
    Determines the 'k' value based on the specified fallback logic.
    Uses the exact logic provided in the requirement description.
    """
    # --- Determine Final Top K ---
    final_k = None
    k_found = False

    # Check rerank_top_k
    rrk_val = row.get('rerank_top_k')
    if pd.notna(rrk_val) and isinstance(rrk_val, (int, float, np.integer, np.floating)) and rrk_val > 0:
        final_k = rrk_val
        k_found = True

    # Check rerank_topk (alternative name) if first not found
    if not k_found:
        rrk_alt_val = row.get('rerank_topk')
        if pd.notna(rrk_alt_val) and isinstance(rrk_alt_val, (int, float, np.integer, np.floating)) and rrk_alt_val > 0:
            final_k = rrk_alt_val
            k_found = True

    # Check fusion_top_k if still not found
    if not k_found:
        fusion_k_val = row.get('fusion_top_k')
        if pd.notna(fusion_k_val) and isinstance(fusion_k_val,
                                                 (int, float, np.integer, np.floating)) and fusion_k_val > 0:
            final_k = fusion_k_val
            k_found = True

    # Check embedding_topk if still not found
    if not k_found:
        embed_k_val = row.get('embedding_topk')
        if pd.notna(embed_k_val) and isinstance(embed_k_val, (int, float, np.integer, np.floating)) and embed_k_val > 0:
            final_k = embed_k_val
            k_found = True

    # Check embedding_top_k if still not found
    if not k_found:
        embed_k_val = row.get('embedding_top_k')
        if pd.notna(embed_k_val) and isinstance(embed_k_val, (
                int, float, np.integer, np.floating)) and embed_k_val > 0:
            final_k = embed_k_val
            k_found = True

    # Check if a K was found
    if not k_found or final_k is None:
        # Instead of printing a warning here, we'll handle NaN K later during data cleaning.
        # Returning NaN allows us to filter rows where K couldn't be determined.
        return np.nan  # Indicate K could not be found

    # Ensure it's an integer
    return int(final_k)


def abbreviate_strategy(strategy_series: pd.Series) -> str:
    """Creates a concise, abbreviated label for a strategy."""
    try:
        method_abbr = ABBREVIATIONS["method"].get(strategy_series['method'], strategy_series['method'][:4])
        embed_abbr = ABBREVIATIONS["embedding"].get(strategy_series['embedding_model_name'],
                                                    strategy_series['embedding_model_name'].split('/')[-1][:6])
        chunk_abbr = ABBREVIATIONS["chunking"].get(strategy_series['chunk_strategy_name'],
                                                   strategy_series['chunk_strategy_name'][:5])

        # Handle potential None/NaN for reranker - map to '<None>' before lookup
        reranker_key = strategy_series['rerank_model_name'] if pd.notna(
            strategy_series['rerank_model_name']) else '<None>'
        rerank_abbr = ABBREVIATIONS["reranker"].get(reranker_key,
                                                    str(reranker_key)[:4])  # Use abbreviated key if not found

        return f"{method_abbr}_{embed_abbr}_{chunk_abbr}_{rerank_abbr}"
    except Exception as e:
        print(f"Warning: Error abbreviating strategy {strategy_series.to_dict()}: {e}")
        # Fallback to concatenating raw values
        return "_".join(map(str, strategy_series[IDENTIFYING_COLS].fillna('NaN')))


def load_and_consolidate_data(csv_paths: list[Path]) -> pd.DataFrame | None:
    """Loads data from multiple CSV files into a single DataFrame."""
    all_dfs = []
    for csv_path in csv_paths:
        if not csv_path.is_file():
            print(f"Error: Results file not found at {csv_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(csv_path)
            # Add source file info if needed later for debugging
            # df['source_file'] = csv_path.name
            all_dfs.append(df)
            print(f"Successfully loaded {csv_path}")
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}. Skipping.")
            continue

    if not all_dfs:
        print("Error: No valid CSV files were loaded.")
        return None

    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Consolidated data from {len(all_dfs)} file(s) into {len(master_df)} rows.")
    return master_df


def get_strategy_groups_from_user(unique_strategies: pd.DataFrame) -> dict[str, list[int]] | None:
    """Identifies unique strategies, asks user for grouping, validates input."""
    print("\n--- Found Unique Strategies ---")
    strategy_map = {}  # Store mapping from ID to original strategy details
    for i, (index, strategy_details) in enumerate(unique_strategies.iterrows()):
        abbreviated_name = abbreviate_strategy(strategy_details)
        print(f"[{i}]: {abbreviated_name}")
        strategy_map[i] = strategy_details.to_dict()  # Store original identifying values

    print("\nPlease provide a dictionary to group strategies for plotting.")
    print("Example: {'Group A': [0, 2], 'Group B': [1, 3]}")
    try:
        user_input_str = input("Enter grouping dictionary: ")
        parsed_input = ast.literal_eval(user_input_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error: Invalid dictionary format. {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: Could not parse input. {e}", file=sys.stderr)
        return None

    # --- Validate Parsed Input ---
    if not isinstance(parsed_input, dict):
        print("Error: Input must be a dictionary.", file=sys.stderr)
        return None

    valid_groups = {}
    max_strategy_id = len(unique_strategies) - 1
    seen_ids = set()

    for group_name, id_list in parsed_input.items():
        if not isinstance(group_name, str) or not group_name:
            print(f"Error: Group names must be non-empty strings. Found: {group_name}", file=sys.stderr)
            return None
        if not isinstance(id_list, list):
            print(f"Error: Value for group '{group_name}' must be a list of integers. Found: {type(id_list)}",
                  file=sys.stderr)
            return None
        if not id_list:
            print(f"Warning: Group '{group_name}' has an empty list of strategy IDs. It will be ignored.")
            continue  # Skip empty groups silently or warn

        processed_ids = []
        for strategy_id in id_list:
            if not isinstance(strategy_id, int):
                print(f"Error: Strategy IDs in group '{group_name}' must be integers. Found: {strategy_id}",
                      file=sys.stderr)
                return None
            if not (0 <= strategy_id <= max_strategy_id):
                print(
                    f"Error: Strategy ID {strategy_id} in group '{group_name}' is out of range (0-{max_strategy_id}).",
                    file=sys.stderr)
                return None
            if strategy_id in seen_ids:
                print(
                    f"Warning: Strategy ID {strategy_id} is included in multiple groups. It will be plotted based on group '{group_name}'.")
                # Decide how to handle duplicates: last one wins, first one wins, or error? Let's allow it but warn.
            processed_ids.append(strategy_id)
            seen_ids.add(strategy_id)

        if processed_ids:  # Only add group if it has valid IDs
            valid_groups[group_name] = processed_ids

    if not valid_groups:
        print("Error: No valid strategy groups were defined.", file=sys.stderr)
        return None

    print("\n--- Using Strategy Groups ---")
    for name, ids in valid_groups.items():
        print(f"'{name}': {ids}")

    return valid_groups, strategy_map


def calculate_f1(row):
    """Safely calculates F1 score, handling potential NaN/zero values."""
    precision = row['precision']
    recall = row['recall']
    # Check for NaN or non-numeric types before calculation
    if pd.isna(precision) or pd.isna(recall) or not isinstance(precision, (int, float, np.number)) or not isinstance(
            recall, (int, float, np.number)):
        return np.nan  # Return NaN if P or R is missing/invalid
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * (precision * recall) / (precision + recall)


def prepare_data_for_plotting(df: pd.DataFrame, selected_groups: dict[str, list[int]],
                              strategy_map: dict[int, dict]) -> pd.DataFrame | None:
    """Filters, prepares metrics, calculates K, and adds grouping info."""

    # Create a mapping from strategy ID back to its identifying dict for easier filtering
    id_to_details = {idx: details for idx, details in strategy_map.items() if
                     idx in [item for sublist in selected_groups.values() for item in sublist]}

    # Filter rows matching selected strategies
    rows_to_keep = []
    # Fill NaN in reranker column for consistent matching during filtering
    df[IDENTIFYING_COLS[3]] = df[IDENTIFYING_COLS[3]].fillna('<None>')

    for strategy_id, details in id_to_details.items():
        # Ensure comparison handles the filled NaN correctly
        details_copy = details.copy()
        if pd.isna(details_copy['rerank_model_name']):
            details_copy['rerank_model_name'] = '<None>'

        # Build filter condition dynamically
        condition = True
        for col in IDENTIFYING_COLS:
            condition &= (df[col] == details_copy[col])

        # Find the group name for this strategy ID
        group_name = None
        for g_name, g_ids in selected_groups.items():
            if strategy_id in g_ids:
                group_name = g_name
                break

        strategy_rows = df[condition].copy()
        if not strategy_rows.empty and group_name:
            strategy_rows['strategy_unique_id'] = strategy_id
            strategy_rows['group_name'] = group_name
            rows_to_keep.append(strategy_rows)

    if not rows_to_keep:
        print("Error: No data rows match the selected strategies.", file=sys.stderr)
        # Revert the fillna if necessary, though maybe not needed if exiting
        # df[IDENTIFYING_COLS[3]] = df[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})
        return None

    plot_df = pd.concat(rows_to_keep, ignore_index=True)
    # Revert the fillna after filtering is done
    plot_df[IDENTIFYING_COLS[3]] = plot_df[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})

    # --- Calculate K Value ---
    plot_df['k'] = plot_df.apply(get_k_value, axis=1)

    # --- Ensure Metrics are Numeric ---
    metric_cols_present = [col for col in ['recall', 'precision', 'f1_score'] if col in plot_df.columns]
    if not any(col in ['recall', 'precision'] for col in metric_cols_present):
        print("Error: DataFrame lacks 'recall' and/or 'precision' columns required for plotting.", file=sys.stderr)
        return None

    for col in metric_cols_present:
        if col != 'f1_score':  # Handle f1 separately
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')

    # --- Calculate F1 Score if Missing ---
    if 'f1_score' not in plot_df.columns:
        if 'precision' in plot_df.columns and 'recall' in plot_df.columns:
            print("Info: 'f1_score' column not found. Calculating from precision and recall...")
            plot_df['f1_score'] = plot_df.apply(calculate_f1, axis=1)
            print("Info: F1 score calculated.")
        else:
            print("Error: Cannot calculate F1 score - 'precision' or 'recall' column missing.", file=sys.stderr)
            # Remove f1_score from metrics to plot if it couldn't be calculated
            if 'f1_score' in METRICS_TO_PLOT: del METRICS_TO_PLOT['f1_score']
    else:
        # Ensure existing f1_score is numeric
        plot_df['f1_score'] = pd.to_numeric(plot_df['f1_score'], errors='coerce')

    # --- Final Cleanup ---
    essential_plot_cols = ['k', 'group_name', 'strategy_unique_id'] + list(METRICS_TO_PLOT.keys())
    initial_rows = len(plot_df)
    plot_df = plot_df.dropna(subset=essential_plot_cols)
    final_rows = len(plot_df)
    if initial_rows != final_rows:
        print(f"Warning: Dropped {initial_rows - final_rows} rows due to missing essential data (K value or metrics).")

    if plot_df.empty:
        print("Error: No valid data points remaining after processing and NaN removal.", file=sys.stderr)
        return None

    # Ensure K is integer after dropna
    plot_df['k'] = plot_df['k'].astype(int)

    print(f"Prepared {len(plot_df)} data points for plotting across {len(selected_groups)} groups.")
    return plot_df


# --- Main Plotting Logic ---

def plot_grouped_results(plot_df: pd.DataFrame, selected_groups: dict[str, list[int]], output_dir: Path,
                         plot_title_base: str):
    """Generates and saves plots for selected metrics, grouped by strategy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Plots in {output_dir} ---")

    group_names = list(selected_groups.keys())
    num_groups = len(group_names)
    # cmap = plt.get_cmap('tab10') # Example colormap
    cmap = plt.get_cmap('viridis', num_groups)  # Colormap based on number of groups

    colors = {group_name: cmap(i) for i, group_name in enumerate(group_names)}

    # --- Plotting Loop Start ---
    for metric, metric_label in METRICS_TO_PLOT.items():
        if metric not in plot_df.columns:
            print(f"Skipping plot for '{metric_label}' as column is missing.")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        group_linestyle_indices = defaultdict(int)  # Track linestyle index per group

        # Iterate through groups first to ensure consistent color/linestyle assignment
        for group_name in group_names:
            group_color = colors[group_name]
            strategy_ids_in_group = selected_groups[group_name]

            # Iterate through unique strategies within the group
            for strategy_id in sorted(strategy_ids_in_group):  # Sort for consistent linestyle assignment
                # Filter data for this specific strategy ID and group
                strategy_df = plot_df[(plot_df['group_name'] == group_name) &
                                      (plot_df['strategy_unique_id'] == strategy_id)].sort_values(by='k')

                if not strategy_df.empty:
                    # Assign linestyle based on the order within the group
                    style_index = group_linestyle_indices[group_name]
                    linestyle = LINESTYLES[style_index % len(LINESTYLES)]
                    group_linestyle_indices[group_name] += 1

                    # Plot the metric vs k for this specific strategy
                    # NO label= here, legend is handled separately
                    ax.plot(
                        strategy_df['k'],
                        strategy_df[metric],
                        # label=f"{group_name} - Strat {strategy_id}", # NO individual labels
                        marker='o',  # Add markers to points
                        linestyle=linestyle,
                        color=group_color
                    )

        ax.set_title(f'{plot_title_base} - {metric_label} vs. K')
        ax.set_xlabel('K (Top-K Results)')
        ax.set_ylabel(metric_label)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Create Custom Legend for Groups ---
        legend_elements = []
        for group_name in group_names:
            legend_elements.append(Line2D([0], [0], color=colors[group_name], lw=2, linestyle='-', label=group_name))

        if legend_elements:
            ax.legend(handles=legend_elements, title="Strategy Groups", bbox_to_anchor=(1.04, 1), loc="upper left")
            # Adjust layout to prevent legend cutoff
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust right margin for legend
        else:
            plt.tight_layout()  # Use default tight layout if no legend

        # --- Save the plot ---
        plot_filename = output_dir / f"{output_dir.name}_{metric}_vs_k.png"
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from one or more results.csv files, allowing strategy grouping.")
    parser.add_argument(
        "results_files",
        type=str,
        nargs='+',  # Accept one or more file paths
        help="Path(s) to the results.csv file(s) containing benchmark data."
    )
    parser.add_argument(
        "-o", "--output-name",
        type=str,
        required=True,
        help="A unique name identifier for this plotting run, used for the output directory name."
    )
    parser.add_argument(
        "-t", "--plot-title",
        type=str,
        required=True,
        help="Base title for the generated plots (metric name will be appended)."
    )

    args = parser.parse_args()

    # --- Determine Project Root and Output Directory ---
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parents[2]  # Adjust if script location differs
    except NameError:
        # If __file__ is not defined (e.g., interactive session), use CWD as project root
        project_root = Path.cwd()
        print(f"Warning: Could not determine script path, assuming project root is CWD: {project_root}")

    output_dir = project_root / "plots" / "performance" / args.output_name

    # --- Main Execution ---
    csv_paths = [Path(f) for f in args.results_files]

    # 1. Load and Consolidate Data
    master_df = load_and_consolidate_data(csv_paths)
    if master_df is None:
        sys.exit(1)  # Exit if loading failed

    # 2. Identify Unique Strategies
    # Handle NaN in reranker temporarily for unique identification
    master_df[IDENTIFYING_COLS[3]] = master_df[IDENTIFYING_COLS[3]].fillna('<None>')
    unique_strategies = master_df[IDENTIFYING_COLS].drop_duplicates().reset_index(drop=True)
    # Revert NaN fill after identification
    master_df[IDENTIFYING_COLS[3]] = master_df[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})
    unique_strategies[IDENTIFYING_COLS[3]] = unique_strategies[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})

    if unique_strategies.empty:
        print("Error: No unique strategies found in the provided data.", file=sys.stderr)
        sys.exit(1)

    # 3. Get Strategy Grouping from User
    user_input_result = get_strategy_groups_from_user(unique_strategies)
    if user_input_result is None:
        sys.exit(1)  # Exit if user input failed
    selected_groups, strategy_map = user_input_result

    # 4. Prepare Data for Plotting
    plot_df = prepare_data_for_plotting(master_df, selected_groups, strategy_map)
    if plot_df is None:
        sys.exit(1)  # Exit if preparation failed

    # 5. Generate and Save Plots
    plot_grouped_results(plot_df, selected_groups, output_dir, args.plot_title)

    print("\nPlotting complete.")
