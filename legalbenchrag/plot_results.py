import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Helper Functions ---


def get_k_value(row):
    """Determines the 'k' value based on the strategy type."""
    # --- Determine Final Rerank K using fallback logic ---
    final_k = None

    # Check rerank_top_k
    rrk_val = row.get('rerank_top_k')
    if pd.notna(rrk_val) and isinstance(rrk_val, (int, float, np.integer, np.floating)) and rrk_val > 0:
        final_k = rrk_val
        return final_k

    # Check rerank_topk (alternative name) if first not found
    if final_k:
        rrk_alt_val = row.get('rerank_topk')
        if pd.notna(rrk_alt_val) and isinstance(rrk_alt_val, (int, float, np.integer, np.floating)) and rrk_alt_val > 0:
            final_k = rrk_alt_val
            return final_k

    # Check fusion_top_k if still not found
    if final_k:
        fusion_k_val = row.get('fusion_top_k')
        if pd.notna(fusion_k_val) and isinstance(fusion_k_val,
                                                 (int, float, np.integer, np.floating)) and fusion_k_val > 0:
            final_k = fusion_k_val
            return final_k

    # Check embedding_topk if still not found
    if final_k:
        embed_k_val = row.get('embedding_topk')
        if pd.notna(embed_k_val) and isinstance(embed_k_val, (int, float, np.integer, np.floating)) and embed_k_val > 0:
            final_k = embed_k_val
            return final_k

    # Check embedding_top_k if still not found
    if final_k:
        embed_k_val = row.get('embedding_top_k')
        if pd.notna(embed_k_val) and isinstance(embed_k_val, (
                int, float, np.integer, np.floating)) and embed_k_val > 0:
            final_k = embed_k_val
            return final_k

    print(f"WARNING: No valid top_k value found for index {row.get('i')}")
    return 64  # Default value if all checks fail


def create_label(row):
    """Creates a concise label for the legend."""
    method = row.get('method', 'unk')
    embed_raw = row.get('embedding_model_name', 'unk_embed')
    chunk_strat = row.get('chunk_strategy_name', 'unk_chunk')
    chunk_size = row.get('chunk_size', '')
    reranker = row.get('rerank_model_company', None) # Use company for baseline

    # Abbreviate method
    method_label = 'Base' if method == 'baseline' else 'HyPA' if method == 'hypa' else method

    # Abbreviate embedding model
    if 'bge-base' in embed_raw:
        embed_label = 'bge-b'
    elif 'bge-large' in embed_raw:
        embed_label = 'bge-l'
    elif 'gte-large' in embed_raw:
        embed_label = 'gte-l'
    elif 'text-embedding-3-small' in embed_raw:
        embed_label = 'oai3-s'
    elif 'text-embedding-3-large' in embed_raw:
        embed_label = 'oai3-l'
    elif 'text-embedding-ada-002' in embed_raw:
        embed_label = 'oai-ada'
    elif 'legal-bert-base' in embed_raw:
        embed_label = 'lbert-b'
    elif 'legal-bert-small' in embed_raw:
        embed_label = 'lbert-s'
    else:
        # Fallback: try to take last part of path/name
        embed_label = embed_raw.split('/')[-1][:10] # Limit length

    # Abbreviate chunk strategy
    chunk_label = f"{chunk_strat[0].upper()}{chunk_size}" if chunk_strat else ''

    # Indicate reranker for Baseline
    rerank_label = ''
    if method == 'baseline' and pd.notna(reranker):
        # Check if reranker is NaN or None before adding label
        rerank_label = f"+{str(reranker)[:3]}"  # e.g., +Coh

    return f"{method_label}_{embed_label}_{chunk_label}{rerank_label}"

# --- Main Plotting Logic ---


def plot_results(results_csv_path: Path):
    """Reads the results CSV and generates plots."""
    if not results_csv_path.is_file():
        print(f"Error: Results file not found at {results_csv_path}")
        return

    try:
        df = pd.read_csv(results_csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # --- Data Preprocessing ---
    # Convert metric columns to numeric FIRST
    metric_cols = ['recall', 'precision', 'f1_score']
    cols_to_convert = []
    if 'recall' in df.columns: cols_to_convert.append('recall')
    if 'precision' in df.columns: cols_to_convert.append('precision')
    # Don't try to convert f1_score yet if it might not exist
    # if 'f1_score' in df.columns: cols_to_convert.append('f1_score')

    if not cols_to_convert:
        print("Error: Cannot find 'recall' or 'precision' columns.")
        return

    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    f1_calculated = False
    if 'f1_score' not in df.columns:
        print("Info: 'f1_score' column not found. Calculating from precision and recall...")
        if 'precision' not in df.columns or 'recall' not in df.columns:
            print("Error: Cannot calculate F1 score - 'precision' or 'recall' column missing.")
            return

        def calculate_f1(row):
            precision = row['precision']
            recall = row['recall']
            # Check for NaN or non-numeric types before calculation
            if pd.isna(precision) or pd.isna(recall) or not isinstance(precision, (int, float)) or not isinstance(
                    recall, (int, float)):
                return np.nan  # Return NaN if P or R is missing/invalid
            if precision + recall == 0:
                return 0.0
            else:
                return 2 * (precision * recall) / (precision + recall)

        df['f1_score'] = df.apply(calculate_f1, axis=1)
        # Ensure the new column is also numeric (it should be, but belt-and-suspenders)
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')
        f1_calculated = True
        print("Info: F1 score calculated.")

        # Save the updated DataFrame back to the CSV file
        try:
            df.to_csv(results_csv_path, index=False)
            print(f"Info: Updated CSV file '{results_csv_path}' with calculated 'f1_score' column.")
        except Exception as e:
            print(f"Warning: Could not update CSV file '{results_csv_path}' with F1 scores. Error: {e}")
            # Continue with plotting using the in-memory DataFrame

    # Now that f1_score definitely exists (or we returned), convert it if it wasn't calculated
    if not f1_calculated and 'f1_score' in df.columns:
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')

    # Drop rows where essential plotting data is missing
    essential_cols = ['method', 'embedding_model_name', 'chunk_strategy_name']
    df = df.dropna(subset=essential_cols)

    # Calculate 'k' and 'label' for each row
    df['k'] = df.apply(get_k_value, axis=1)
    df['label'] = df.apply(create_label, axis=1)

    # Drop rows where k could not be determined or metrics are NaN
    # Ensure f1_score is included in the check
    df = df.dropna(subset=['k', 'recall', 'precision', 'f1_score'])
    if not df.empty:  # Check if DataFrame is not empty after dropna
        df['k'] = df['k'].astype(int)  # Ensure k is integer type
    else:
        print("No valid data points remaining after handling NaNs.")
        return

    # Get unique labels (methods) to plot
    unique_labels = df['label'].unique()
    if len(unique_labels) == 0:
        print("No valid data points found to plot after processing.")
        return

    print(f"Found {len(unique_labels)} unique strategy configurations to plot.")

    # --- Plotting ---
    plot_dir = Path("./plots")
    plot_dir.mkdir(exist_ok=True)
    run_name = results_csv_path.parent.name  # Extract run name from parent dir

    # Define metrics to plot
    metrics_to_plot = {
        'precision': 'Precision vs. K',
        'recall': 'Recall vs. K',
        'f1_score': 'F1-Score vs. K'
    }

    # Generate a color map for unique labels
    cmap = plt.get_cmap('tab10')  # Get the colormap object
    colors_list = cmap.colors  # Get the list of discrete colors from the map

    # --- Plotting Loop Start ---
    for metric, title in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, label in enumerate(sorted(unique_labels)):  # Sort labels for consistent color assignment
            # Filter data for the current method label
            method_df = df[df['label'] == label].sort_values(by='k')

            if not method_df.empty:
                # Assign color using modulo operator to cycle through the discrete color list
                color = colors_list[i % len(colors_list)]
                # Plot the metric vs k for this method
                ax.plot(
                    method_df['k'],
                    method_df[metric],
                    label=label,
                    marker='o',  # Add markers to points
                    linestyle='-',
                    color=color
                )

        ax.set_title(f'{title} ({run_name})')
        ax.set_xlabel('K (Top-K Results)')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Place legend outside of the plot area
        ax.legend(title="Methods", bbox_to_anchor=(1.04, 1), loc="upper left")

        # Adjust layout to prevent legend cutoff
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right margin for legend

        # Save the plot
        plot_filename = plot_dir / f"{run_name}_{metric}_vs_k.png"
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from results_summary.csv.")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the specific results_summary.csv file to plot."
    )

    args = parser.parse_args()
    results_path = Path(args.results_file)
    plot_results(results_path)
