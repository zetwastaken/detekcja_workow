"""Utilities for loading and parsing YOLO training run results."""

from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd


def load_training_results(run_folder: Path) -> Optional[Dict]:
    """
    Load training results from a run folder.

    Returns a dict with metrics for the best epoch by mAP50-95,
    or None if the results file is missing/invalid.
    """
    results_file = Path(run_folder) / "results.csv"

    try:
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()

        map_col = next((c for c in df.columns if "mAP50-95" in c), None)
        if map_col is None:
            print(f"Warning: Could not find mAP50-95 column in {run_folder}")
            return None

        df = df.dropna(subset=[map_col])
        if df.empty:
            return None

        best_idx = df[map_col].idxmax()
        best = df.loc[best_idx]

        def get_metric(df_row, pattern: str):
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return float(df_row[col])
            return np.nan

        return {
            "run_folder": Path(run_folder).name,
            "best_epoch": int(best.get("epoch", best_idx)),
            "total_epochs": len(df),
            "mAP50-95": get_metric(best, "mAP50-95"),
            "mAP50": get_metric(best, "mAP50("),
            "precision": get_metric(best, "precision"),
            "recall": get_metric(best, "recall"),
            "results_df": df,
        }
    except Exception as e:
        print(f"Error loading results from {run_folder}: {e}")
        return None
