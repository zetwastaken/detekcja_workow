"""
Comparison script for analyzing results from all depth model training runs.
Generates comparative tables and visualizations to identify the best performing depth model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs" / "segment"
OUTPUT_DIR = PROJECT_ROOT / "comparison_results"


def find_depth_training_runs() -> dict:
    """
    Find all depth model training runs in runs/segment/.
    Returns dict mapping model_name to run_folder path.
    """
    runs = {}

    for run_folder in RUNS_DIR.iterdir():
        if run_folder.is_dir() and run_folder.name.startswith("depth_"):
            results_file = run_folder / "results.csv"
            if results_file.exists():
                # Extract model name from run folder name
                # Format: depth_<model_name>_<timestamp>
                parts = run_folder.name.split("_")
                if len(parts) >= 3:
                    # Reconstruct model name (may contain underscores)
                    model_name = "_".join(
                        parts[1:-2]
                    )  # Remove 'depth_' prefix and timestamp

                    # If multiple runs for same model, keep the latest
                    if model_name in runs:
                        # Compare timestamps
                        existing_ts = runs[model_name].name.split("_")[-2:]
                        new_ts = parts[-2:]
                        if new_ts > existing_ts:
                            runs[model_name] = run_folder
                    else:
                        runs[model_name] = run_folder

    return runs


def find_rgb_baseline() -> Path:
    """Find RGB baseline training run if it exists."""
    for run_folder in RUNS_DIR.iterdir():
        if run_folder.is_dir() and "rgb_baseline" in run_folder.name:
            results_file = run_folder / "results.csv"
            if results_file.exists():
                return run_folder
    return None


def load_training_results(run_folder: Path) -> dict:
    """
    Load training results from a run folder.
    Returns dict with best epoch metrics.
    """
    results_file = run_folder / "results.csv"

    try:
        df = pd.read_csv(results_file)
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Find best epoch by mAP50-95 (for segmentation, column is 'metrics/mAP50-95(M)')
        # Try different column name formats
        map_col = None
        for col in df.columns:
            if "mAP50-95" in col:
                map_col = col
                break

        if map_col is None:
            print(f"Warning: Could not find mAP50-95 column in {run_folder.name}")
            return None

        df = df.dropna(subset=[map_col])

        if df.empty:
            return None

        best_idx = df[map_col].idxmax()
        best = df.loc[best_idx]

        # Extract metrics with flexible column names
        def get_metric(df_row, pattern):
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return float(df_row[col])
            return np.nan

        return {
            "run_folder": run_folder.name,
            "best_epoch": int(best.get("epoch", best_idx)),
            "total_epochs": len(df),
            "mAP50-95": get_metric(best, "mAP50-95"),
            "mAP50": get_metric(best, "mAP50("),
            "precision": get_metric(best, "precision"),
            "recall": get_metric(best, "recall"),
            "results_df": df,  # Keep full dataframe for plotting
        }
    except Exception as e:
        print(f"Error loading results from {run_folder}: {e}")
        return None


def compare_all_models(include_rgb: bool = True) -> pd.DataFrame:
    """
    Compare all depth model training results.
    Returns DataFrame with comparison metrics.
    """
    print("\n" + "=" * 90)
    print("DEPTH MODEL COMPARISON - YOLOV8 SEGMENTATION")
    print("=" * 90)

    # Find all depth training runs
    depth_runs = find_depth_training_runs()

    if not depth_runs:
        print("No depth model training runs found!")
        print(f"Run train_all_depth_models.py first to train models")
        return None

    print(f"\nFound {len(depth_runs)} depth model runs:")
    for name, path in sorted(depth_runs.items()):
        print(f"  - {name}: {path.name}")

    # Load results for each model
    results = []

    for model_name, run_folder in sorted(depth_runs.items()):
        metrics = load_training_results(run_folder)
        if metrics:
            metrics["model_name"] = model_name
            metrics["model_type"] = "depth"
            results.append(metrics)

    # Add RGB baseline if requested
    if include_rgb:
        rgb_run = find_rgb_baseline()
        if rgb_run:
            metrics = load_training_results(rgb_run)
            if metrics:
                metrics["model_name"] = "rgb_baseline"
                metrics["model_type"] = "rgb"
                results.append(metrics)
            print(f"  - RGB baseline: {rgb_run.name}")

    if not results:
        print("No valid results found!")
        return None

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(
        [
            {
                "Model": r["model_name"],
                "Type": r["model_type"],
                "mAP50-95": r["mAP50-95"],
                "mAP50": r["mAP50"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "Best Epoch": r["best_epoch"],
                "Total Epochs": r["total_epochs"],
                "Run": r["run_folder"],
            }
            for r in results
        ]
    )

    # Sort by mAP50-95
    comparison_df = comparison_df.sort_values("mAP50-95", ascending=False)

    return comparison_df, results


def print_ranking(df: pd.DataFrame):
    """Print ranked comparison table."""
    print("\n" + "=" * 90)
    print("📊 MODEL RANKING (by mAP50-95)")
    print("=" * 90 + "\n")

    medals = ["🥇", "🥈", "🥉"] + ["  "] * (len(df) - 3)

    for idx, (_, row) in enumerate(df.iterrows()):
        medal = medals[idx] if idx < len(medals) else "  "
        model_type_icon = "🎨" if row["Type"] == "depth" else "📷"

        print(
            f"{medal} {model_type_icon} {row['Model']:25} | "
            f"mAP50-95={row['mAP50-95']:.4f} | "
            f"mAP50={row['mAP50']:.4f} | "
            f"P={row['Precision']:.4f} | "
            f"R={row['Recall']:.4f} | "
            f"Best@{row['Best Epoch']}/{row['Total Epochs']}"
        )

    print("\n" + "=" * 90)

    # Winner announcement
    best = df.iloc[0]
    print(f"🏆 BEST MODEL: {best['Model']}")
    print(f"   Type: {'Depth Map' if best['Type'] == 'depth' else 'RGB'}")
    print(f"   mAP50-95: {best['mAP50-95']:.4f}")
    print(f"   mAP50: {best['mAP50']:.4f}")
    print("=" * 90)

    # Depth vs RGB comparison
    depth_models = df[df["Type"] == "depth"]
    rgb_models = df[df["Type"] == "rgb"]

    if not depth_models.empty and not rgb_models.empty:
        best_depth = depth_models.iloc[0]
        rgb_baseline = rgb_models.iloc[0]

        improvement = (
            (best_depth["mAP50-95"] - rgb_baseline["mAP50-95"])
            / rgb_baseline["mAP50-95"]
        ) * 100

        print(f"\n📈 DEPTH vs RGB COMPARISON:")
        print(
            f"   Best Depth Model: {best_depth['Model']} (mAP50-95: {best_depth['mAP50-95']:.4f})"
        )
        print(
            f"   RGB Baseline:     {rgb_baseline['Model']} (mAP50-95: {rgb_baseline['mAP50-95']:.4f})"
        )
        print(f"   Improvement: {improvement:+.2f}%")


def plot_comparison(df: pd.DataFrame, results: list, output_dir: Path):
    """Generate comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")

    # 1. Bar chart of mAP50-95
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ["#2ecc71" if t == "depth" else "#3498db" for t in df["Type"]]
    bars = ax.barh(
        df["Model"], df["mAP50-95"], color=colors, edgecolor="white", linewidth=0.5
    )

    ax.set_xlabel("mAP50-95", fontsize=12)
    ax.set_title("Depth Model Comparison - mAP50-95", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, df["mAP50-95"]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Depth Models"),
        Patch(facecolor="#3498db", label="RGB Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_mAP50-95.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["mAP50-95", "mAP50", "Precision", "Recall"]

    for ax, metric in zip(axes.flatten(), metrics):
        colors = ["#2ecc71" if t == "depth" else "#3498db" for t in df["Type"]]
        bars = ax.barh(df["Model"], df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.invert_yaxis()

        for bar, val in zip(bars, df[metric]):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=8,
            )

    plt.suptitle("Depth Model Comparison - All Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Training curves comparison (if we have full dataframes)
    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        if "results_df" in r:
            df_train = r["results_df"]
            # Find mAP50-95 column
            map_col = [c for c in df_train.columns if "mAP50-95" in c]
            if map_col:
                ax.plot(
                    df_train["epoch"],
                    df_train[map_col[0]],
                    label=r["model_name"],
                    linewidth=1.5,
                    alpha=0.8,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP50-95", fontsize=12)
    ax.set_title(
        "Training Curves - mAP50-95 Over Epochs", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Visualizations saved to: {output_dir}")


def export_results(df: pd.DataFrame, output_dir: Path):
    """Export comparison results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"comparison_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Results exported to: {csv_path}")


def main(include_rgb: bool = True, export: bool = True, plot: bool = True):
    """
    Main comparison function.

    Args:
        include_rgb: Include RGB baseline in comparison
        export: Export results to CSV
        plot: Generate comparison plots
    """
    result = compare_all_models(include_rgb)

    if result is None:
        return

    comparison_df, full_results = result

    # Print ranking
    print_ranking(comparison_df)

    # Export and visualize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if export:
        export_results(comparison_df, OUTPUT_DIR)

    if plot:
        plot_comparison(comparison_df, full_results, OUTPUT_DIR)

    return comparison_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare all depth model training results"
    )
    parser.add_argument(
        "--no-rgb", action="store_true", help="Exclude RGB baseline from comparison"
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Don't export results to CSV"
    )
    parser.add_argument("--no-plot", action="store_true", help="Don't generate plots")

    args = parser.parse_args()

    main(include_rgb=not args.no_rgb, export=not args.no_export, plot=not args.no_plot)
