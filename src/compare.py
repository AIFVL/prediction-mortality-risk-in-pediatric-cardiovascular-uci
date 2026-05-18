from pathlib import Path
import pandas as pd

from src.config import RESULTS_DIR
from src.io_utils import ensure_dir, read_csv, write_csv


def run_comparison() -> Path:
    ensure_dir(RESULTS_DIR)

    latest_path = RESULTS_DIR / "evaluation_summary_latest.csv"
    if latest_path.exists():
        combined = read_csv(latest_path)
        combined_path = RESULTS_DIR / "model_comparison.csv"
        write_csv(combined, combined_path)
        return combined_path

    summary_files = sorted(RESULTS_DIR.glob("evaluation_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError("No evaluation_summary files found in results directory.")

    dfs = [read_csv(path) for path in summary_files]
    combined = pd.concat(dfs, ignore_index=True)

    combined_path = RESULTS_DIR / "model_comparison.csv"
    write_csv(combined, combined_path)

    return combined_path
