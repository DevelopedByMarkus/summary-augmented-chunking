import os
import pandas as pd
from pathlib import Path

# --- Try to import the analysis helper (works whether it's a sibling module or package) ---
try:
    from analyze_retrieval import analyze_filepaths_for_plotting  # same dir import
except Exception:
    try:
        from .analyze_retrieval import analyze_filepaths_for_plotting  # package-style
    except Exception as e:
        analyze_filepaths_for_plotting = None
        _ANALYZE_IMPORT_ERR = e


def _aggregate_ide_from_json(dir_path: Path) -> float:
    """
    Compute weighted Inter-Document Error (IDE %) across all JSON retrieval files in dir_path.
    IDE (%) = 100 * (sum incorrect) / (sum correct + sum incorrect)
    Returns float('nan') if no usable files are found.
    """
    if analyze_filepaths_for_plotting is None:
        raise ImportError(
            "Could not import 'analyze_filepaths_for_plotting' from analyze_retrieval.py. "
            f"Original error: {_ANALYZE_IMPORT_ERR}"
        )

    total_correct = 0
    total_incorrect = 0

    for fp in dir_path.glob("*.json"):
        try:
            res = analyze_filepaths_for_plotting(str(fp))
        except Exception:
            # Skip files that fail to parse/analyze
            continue

        if not res or "overall" not in res:
            continue

        overall = res["overall"]
        total_correct += int(overall.get("correct", 0))
        total_incorrect += int(overall.get("incorrect", 0))

    denom = total_correct + total_incorrect
    if denom == 0:
        return float("nan")

    return (total_incorrect / denom) * 100.0


def calc_avg_precision_recall(folder_path: str):
    # --- Setup Paths ---
    project_root = Path.cwd()
    run_dir = project_root / "results" / "legalbenchrag" / folder_path

    # --- Read results CSV ---
    file_path = run_dir / "results_summary.csv"
    df = pd.read_csv(file_path)

    # --- Check required columns ---
    if "precision" not in df.columns or "recall" not in df.columns:
        raise ValueError("CSV muss Spalten 'precision' und 'recall' enthalten.")

    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()

    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")

    # --- Compute IDE from retrieval JSON files (if available) ---
    try:
        ide_pct = _aggregate_ide_from_json(run_dir)
        if pd.notna(ide_pct):
            print(f"Average IDE (retrieval): {ide_pct:.4f}%")
        else:
            print("Average IDE (retrieval): n/a (no usable retrieval JSON files found)")
    except ImportError as e:
        print(f"Average IDE (retrieval): n/a ({e})")


if __name__ == "__main__":
    folder = "_2025-08-11_10-38-01_hybrid_s-rcts-500+150_gteL_w05_m25"
    calc_avg_precision_recall(folder)
