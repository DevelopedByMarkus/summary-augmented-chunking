import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.lines import Line2D
import sys
from collections import defaultdict
import re
from sac_rag.utils.abbreviations import ABBREVIATIONS

DEFAULT_ABBR = "unk"
NONE_ABBR = "X"  # Placeholder for when a component (like reranker) is not used


# --- Constants ---

IDENTIFYING_COLS = ['method', 'embedding_model_name', 'chunk_strategy_name', 'rerank_model_name', 'config_file']
METRICS_TO_PLOT = {'precision': 'Precision', 'recall': 'Recall', 'f1_score': 'F1-Score'}
LINESTYLES = ['-', '--', '-.', ':']
DEFAULT_K = 64


# --- Helper Functions ---


def get_abbreviation(value: str | None, category: str) -> str:
    """Gets the abbreviation for a given value and category."""
    if value is None and category == "reranker":
        return NONE_ABBR  # Special handling for optional reranker

    category_map = ABBREVIATIONS.get(category)
    if not category_map:
        print(f"Warning: Unknown abbreviation category '{category}'. Using default '{DEFAULT_ABBR}'.")
        return DEFAULT_ABBR

    abbr = category_map.get(value)  # type: ignore
    if abbr is None:
        # Check again explicitly for None in case it wasn't the reranker category
        if value is None:
            return NONE_ABBR
        # Value is not None, but not found in map
        print(f"Warning: Unknown value '{value}' for category '{category}'. Using default '{DEFAULT_ABBR}'.")
        return DEFAULT_ABBR
    return abbr


def get_k_value(row):
    """
    Determines the 'k' value based on the specified fallback logic.
    """
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

    # Check embedding_top_k (Corrected from embedding_topk)
    if not k_found:
        embed_k_val = row.get('embedding_top_k')
        if pd.notna(embed_k_val) and isinstance(embed_k_val, (
                int, float, np.integer, np.floating)) and embed_k_val > 0:
            final_k = embed_k_val
            k_found = True

    if not k_found or final_k is None:
        return np.nan

    return int(final_k)


def abbreviate_strategy(strategy_series: pd.Series) -> str:
    """Creates a concise, abbreviated label for a strategy."""
    try:
        config_path = strategy_series["config_file"]
        # Remove prefix and suffix
        return config_path.split("./configs/")[1].replace(".json", "")
        # method_abbr = ABBREVIATIONS["method"].get(strategy_series['method'], strategy_series['method'][:4])
        # embed_abbr = ABBREVIATIONS["embedding"].get(strategy_series['embedding_model_name'], strategy_series['embedding_model_name'].split('/')[-1][:6])
        # chunk_abbr = ABBREVIATIONS["chunking"].get(strategy_series['chunk_strategy_name'], strategy_series['chunk_strategy_name'][:5])
        # reranker_key = strategy_series['rerank_model_name'] if pd.notna(strategy_series['rerank_model_name']) else '<None>'
        # rerank_abbr = ABBREVIATIONS["reranker"].get(reranker_key, str(reranker_key)[:4])
        # return f"{method_abbr}_{chunk_abbr}_{embed_abbr}_{rerank_abbr}"
    except Exception as e:
        print(f"Warning: Error abbreviating strategy {strategy_series.to_dict()}: {e}")
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


def get_strategy_groups_from_user(unique_strategies: pd.DataFrame):
    """Identifies unique strategies, asks user for grouping, validates input."""
    print("\n--- Found Unique Strategies ---")
    strategy_map = {}
    for i, (_, strategy_details) in enumerate(unique_strategies.iterrows()):
        abbreviated_name = abbreviate_strategy(strategy_details)
        print(f"[{i}]: {abbreviated_name}")
        strategy_map[i] = strategy_details.to_dict()

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
            continue

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
            processed_ids.append(strategy_id)
            seen_ids.add(strategy_id)

        if processed_ids:
            valid_groups[group_name] = processed_ids

    if not valid_groups:
        print("Error: No valid strategy groups were defined.", file=sys.stderr)
        return None

    print("\n--- Using Strategy Groups ---")
    for name, ids in valid_groups.items():
        print(f"'{name}': {ids}")

    return valid_groups, strategy_map


def detect_and_select_tasks(df: pd.DataFrame) -> list[str]:
    """
    Detects available benchmark tasks from DataFrame columns and prompts the user to select which ones to plot.
    """
    found_tasks = set()
    # Regex to find prefixes like 'maud|' or 'cuad|'
    task_regex = re.compile(r"(\w+)\|precision")

    for col in df.columns:
        match = task_regex.match(col)
        if match:
            found_tasks.add(match.group(1))

    # Check for the default 'Overall' task (non-prefixed metrics)
    if 'precision' in df.columns and 'recall' in df.columns:
        # Add "Overall" to the beginning for consistent ordering
        available_tasks = ["Overall"] + sorted(list(found_tasks))
    else:
        available_tasks = sorted(list(found_tasks))

    if not available_tasks:
        print("Error: No plottable tasks found. The script requires 'precision'/'recall' columns or "
              "prefixed columns like 'maud|precision'.", file=sys.stderr)
        return []

    print("\n--- Found Plottable Tasks ---")
    for i, task_name in enumerate(available_tasks):
        print(f"[{i}]: {task_name}")

    while True:
        try:
            user_input = input("Enter the indices of tasks to plot (e.g., '0' or '0, 1, 3' -> without '): ")
            selected_indices = [int(i.strip()) for i in user_input.split(',')]

            # Validate indices
            if all(0 <= i < len(available_tasks) for i in selected_indices):
                selected_tasks = [available_tasks[i] for i in selected_indices]
                print(f"--- Selected tasks: {selected_tasks} ---")
                return selected_tasks
            else:
                print(f"Error: Invalid index. Please enter indices between 0 and {len(available_tasks) - 1}.",
                      file=sys.stderr)
        except (ValueError, IndexError):
            print("Error: Invalid input. Please enter a comma-separated list of valid integers.", file=sys.stderr)


def ask_for_plotting_mode() -> bool:
    """Asks the user if they want to combine multi-task plots."""
    while True:
        choice = input("\nPlot selected tasks on the same graph? (yes/no): ").lower().strip()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def calculate_f1(precision: pd.Series, recall: pd.Series) -> pd.Series:
    """Safely calculates F1 score from precision and recall Series."""
    # Ensure inputs are numeric
    precision = pd.to_numeric(precision, errors='coerce')
    recall = pd.to_numeric(recall, errors='coerce')

    # Calculate F1, handling division by zero
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1.fillna(0)  # If P+R is 0 or NaN, F1 is 0


def prepare_data_for_task(df: pd.DataFrame, selected_groups: dict[str, list[int]],
                          strategy_map: dict[int, dict], task_name: str) -> pd.DataFrame | None:
    """
    Filters and prepares data for a SINGLE specified task.
    This involves selecting the correct prefixed columns and renaming them for the plotting function.
    """
    # Determine the column prefix for the given task
    prefix = '' if task_name == 'Overall' else f'{task_name}|'
    metric_cols = {
        'precision': f'{prefix}precision',
        'recall': f'{prefix}recall',
        'f1_score': f'{prefix}f1_score'
    }

    # Check if required source columns exist
    if metric_cols['precision'] not in df.columns or metric_cols['recall'] not in df.columns:
        print(f"Warning: Skipping task '{task_name}' because required columns "
              f"('{metric_cols['precision']}', '{metric_cols['recall']}') were not found.")
        return None

    # Create a mapping from strategy ID back to its identifying dict for easier filtering
    id_to_details = {idx: details for idx, details in strategy_map.items() if
                     idx in [item for sublist in selected_groups.values() for item in sublist]}
    # Get all selected strategy IDs (flatten the list of lists from selected_groups)
    selected_ids = [
        strategy_id
        for group_ids in selected_groups.values()
        for strategy_id in group_ids
    ]

    # Build a mapping of strategy ID -> details for only the selected strategies
    id_to_details = {
        strategy_id: details
        for strategy_id, details in strategy_map.items()
        if strategy_id in selected_ids
    }

    # Filter rows matching selected strategies
    rows_to_keep = []
    df[IDENTIFYING_COLS[3]] = df[IDENTIFYING_COLS[3]].fillna('<None>')  # Handle NaN in reranker for matching

    for strategy_id, details in id_to_details.items():
        details_copy = details.copy()
        if pd.isna(details_copy['rerank_model_name']):
            details_copy['rerank_model_name'] = '<None>'

        condition = pd.Series([True] * len(df))
        for col in IDENTIFYING_COLS:
            condition &= (df[col] == details_copy[col])

        group_name = next((g_name for g_name, g_ids in selected_groups.items() if strategy_id in g_ids), None)

        strategy_rows = df[condition].copy()
        if not strategy_rows.empty and group_name:
            strategy_rows['strategy_unique_id'] = strategy_id
            strategy_rows['group_name'] = group_name
            rows_to_keep.append(strategy_rows)

    if not rows_to_keep:
        print("Error: No data rows match the selected strategies.", file=sys.stderr)
        df[IDENTIFYING_COLS[3]] = df[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})  # Revert NaN fill
        return None

    plot_df = pd.concat(rows_to_keep, ignore_index=True)
    plot_df[IDENTIFYING_COLS[3]] = plot_df[IDENTIFYING_COLS[3]].replace({'<None>': np.nan})  # Revert NaN fill

    # --- Standardize Metric Columns ---
    # Rename the task-specific columns to the generic names for plotting
    plot_df['precision'] = pd.to_numeric(plot_df[metric_cols['precision']], errors='coerce')
    plot_df['recall'] = pd.to_numeric(plot_df[metric_cols['recall']], errors='coerce')

    # Calculate F1 score if missing, otherwise use existing
    if metric_cols['f1_score'] in plot_df.columns:
        plot_df['f1_score'] = pd.to_numeric(plot_df[metric_cols['f1_score']], errors='coerce')
    else:
        print(f"Info: Calculating F1 score for task '{task_name}'.")
        plot_df['f1_score'] = calculate_f1(plot_df['precision'], plot_df['recall'])

    # --- Calculate K Value and Final Cleanup ---
    plot_df['k'] = plot_df.apply(get_k_value, axis=1)

    essential_plot_cols = ['k', 'group_name', 'strategy_unique_id'] + list(METRICS_TO_PLOT.keys())
    initial_rows = len(plot_df)
    plot_df.dropna(subset=essential_plot_cols, inplace=True)
    final_rows = len(plot_df)
    if initial_rows != final_rows:
        print(
            f"Warning: For task '{task_name}', dropped {initial_rows - final_rows} rows due to missing essential data.")

    if plot_df.empty:
        print(f"Error: No valid data points remaining for task '{task_name}' after processing.", file=sys.stderr)
        return None

    plot_df['k'] = plot_df['k'].astype(int)
    print(f"Prepared {len(plot_df)} data points for task '{task_name}'.")
    return plot_df


# --- Main Plotting Logic ---

def plot_grouped_results(plot_df: pd.DataFrame, selected_groups: dict[str, list[int]], output_dir: Path,
                         plot_title_base: str, task_name: str):
    """
    Generates and saves plots for a single task, grouping by strategy.
    Strategy groups are mapped to COLOR, and strategies within groups are mapped to LINESTYLE.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Plots for Task '{task_name}' in {output_dir} ---")

    group_names = list(selected_groups.keys())
    cmap = plt.get_cmap('viridis', len(group_names))
    colors = {group_name: cmap(i) for i, group_name in enumerate(group_names)}

    for metric, metric_label in METRICS_TO_PLOT.items():
        if metric not in plot_df.columns:
            print(f"Skipping plot for '{metric_label}' as column is missing for task '{task_name}'.")
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        group_linestyle_indices = defaultdict(int)

        for group_name in group_names:
            group_color = colors[group_name]
            strategy_ids_in_group = selected_groups[group_name]

            for strategy_id in sorted(strategy_ids_in_group):
                strategy_df = plot_df[(plot_df['group_name'] == group_name) &
                                      (plot_df['strategy_unique_id'] == strategy_id)].sort_values(by='k')

                if not strategy_df.empty:
                    style_index = group_linestyle_indices[group_name]
                    linestyle = LINESTYLES[style_index % len(LINESTYLES)]
                    group_linestyle_indices[group_name] += 1

                    label = f"{group_name} (Strat {strategy_id})"  # More descriptive label
                    ax.plot(strategy_df['k'], strategy_df[metric], marker='o', linestyle=linestyle, color=group_color,
                            label=label)

        # Fix y-axis scale per metric
        if 'precision' in metric.lower() or 'f1_score' in metric.lower():
            ax.set_ylim(0, 0.4)
        elif 'recall' in metric:
            ax.set_ylim(0, 0.8)
        else:
            print("No precision, recall or f1_score was found in metric. No ax.set_ylim possible. metric: " + metric)

        task_display_name = "Overall" if task_name == "Overall" else task_name.upper()
        ax.set_title(f'{plot_title_base}: {task_display_name} {metric_label} vs. K', fontsize=18)
        ax.set_xlabel('Top-K', fontsize=17)
        ax.set_ylabel(metric_label, fontsize=17)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title="Strategy Groups", fontsize=15)
        plt.tight_layout()

        metric_filename = metric.replace("|", "_")
        task_filename = "overall" if task_name == "Overall" else task_name.lower()
        plot_filename = output_dir / f"{output_dir.name}_{task_filename}_{metric_filename}_vs_topk.png"
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)


def plot_combined_results(master_df: pd.DataFrame, selected_groups: dict[str, list[int]],
                          strategy_map: dict[int, dict], selected_tasks: list[str],
                          output_dir: Path, plot_title_base: str):
    """
    Generates a single plot combining multiple tasks.
    Tasks are mapped to COLOR, strategy groups are mapped to LINESTYLE.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Combined Plots for Tasks {selected_tasks} in {output_dir} ---")

    # --- Prepare Color/Style Mappings ---
    task_cmap = plt.get_cmap('plasma', len(selected_tasks))
    task_colors = {task: task_cmap(i) for i, task in enumerate(selected_tasks)}

    group_names = list(selected_groups.keys())
    group_linestyles = {group: LINESTYLES[i % len(LINESTYLES)] for i, group in enumerate(group_names)}

    # Loop through each metric to create a separate plot file for it
    for metric, metric_label in METRICS_TO_PLOT.items():
        fig, ax = plt.subplots(figsize=(14, 9))

        # --- Pre-calculate K values for the entire dataframe once ---
        master_df['k'] = master_df.apply(get_k_value, axis=1)

        # Plot lines for each combination of task and strategy
        for task_name in selected_tasks:
            prefix = '' if task_name == 'Overall' else f'{task_name}|'
            metric_col = f"{prefix}{metric}"

            if metric_col not in master_df.columns:
                print(
                    f"Warning: Skipping task '{task_name}' in combined plot for '{metric_label}' due to missing column '{metric_col}'.")
                continue

            task_color = task_colors[task_name]

            for group_name in group_names:
                group_linestyle = group_linestyles[group_name]

                for strategy_id in selected_groups[group_name]:
                    details = strategy_map[strategy_id]

                    # Build filter condition
                    condition = pd.Series([True] * len(master_df))
                    for col, val in details.items():
                        if pd.isna(val):
                            condition &= master_df[col].isna()
                        else:
                            condition &= (master_df[col] == val)

                    strategy_df = master_df[condition].copy()
                    strategy_df[metric_col] = pd.to_numeric(strategy_df[metric_col], errors='coerce')
                    strategy_df = strategy_df.dropna(subset=['k', metric_col]).sort_values(by='k')

                    if not strategy_df.empty:
                        ax.plot(strategy_df['k'], strategy_df[metric_col], marker='o', markersize=4,
                                color=task_color, linestyle=group_linestyle)

        ax.set_title(f'Combined Plot: {metric_label} vs. K', fontsize=18)
        ax.set_xlabel('Top-K', fontsize=17)
        ax.set_ylabel(metric_label, fontsize=17)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Fix y-axis scale per metric
        if 'precision' in metric.lower() or 'f1_score' in metric.lower():
            ax.set_ylim(0, 0.4)
        elif 'recall' in metric:
            ax.set_ylim(0, 0.8)
        else:
            print("No precision, recall or f1_score was found in metric. No ax.set_ylim possible. metric: " + metric)

        # --- Create Custom Legends ---
        task_legend_elements = [Line2D([0], [0], color=task_colors[task], lw=3, label=task) for task in selected_tasks
                                if f"{'' if task == 'Overall' else task + '|'}{metric}" in master_df.columns]
        group_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=group_linestyles[group], label=group)
                                 for group in group_names]

        if task_legend_elements and group_legend_elements:
            legend1 = ax.legend(handles=task_legend_elements, title="Tasks", loc='upper left', bbox_to_anchor=(1.02, 1),
                                borderaxespad=0.)
            ax.add_artist(legend1)
            ax.legend(handles=group_legend_elements, title="Strategy Groups", loc='upper left',
                      bbox_to_anchor=(1.02, 0.7), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # --- Save the plot ---
        metric_filename = metric.replace("|", "_")
        plot_filename = output_dir / f"{output_dir.name}_combined_{metric_filename}_vs_topk.png"
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved combined plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from CSV files with interactive task and strategy selection.")
    parser.add_argument(
        "--results_files", type=str, nargs='+', help="Path(s) to the results.csv file(s)."
    )
    parser.add_argument(
        "-o", "--output-name", type=str, required=True, help="A unique name for the output directory."
    )
    parser.add_argument(
        "-t", "--plot-title", type=str, required=True, help="Base title for the plots."
    )
    args = parser.parse_args()

    # --- Setup Paths ---
    project_root = Path.cwd()
    output_dir = project_root / "plots" / "legalbenchrag" / "performance" / args.output_name

    # --- Main Execution Flow ---
    # 1. Load Data
    master_df = load_and_consolidate_data([Path(f) for f in args.results_files])
    if master_df is None:
        sys.exit(1)

    # 2. Identify Unique Strategies and Get User Grouping
    temp_df = master_df.copy()
    temp_df[IDENTIFYING_COLS[3]] = temp_df[IDENTIFYING_COLS[3]].fillna('<None>')
    unique_strategies = temp_df[IDENTIFYING_COLS].drop_duplicates().reset_index(drop=True)
    if unique_strategies.empty:
        print("Error: No unique strategies found.", file=sys.stderr)
        sys.exit(1)

    user_input_result = get_strategy_groups_from_user(unique_strategies)
    if user_input_result is None:
        sys.exit(1)
    selected_groups, strategy_map = user_input_result

    # 3. Detect and Select Tasks to Plot
    selected_tasks = detect_and_select_tasks(master_df)
    if not selected_tasks:
        sys.exit(1)

    # 4. Determine Plotting Mode and Generate Plots
    combine_plots = False
    if len(selected_tasks) > 1:
        combine_plots = ask_for_plotting_mode()

    if combine_plots:
        # Combined plotting mode
        plot_combined_results(master_df, selected_groups, strategy_map, selected_tasks, output_dir, args.plot_title)
    else:
        # Separate plotting mode
        for task in selected_tasks:
            plot_df = prepare_data_for_task(master_df, selected_groups, strategy_map, task)
            if plot_df is not None and not plot_df.empty:
                plot_grouped_results(plot_df, selected_groups, output_dir, args.plot_title, task_name=task)
            else:
                print(f"Skipping plot generation for task '{task}' as no valid data was prepared.")

    print("\nPlotting complete.")
