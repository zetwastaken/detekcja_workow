"""
Unified comparison script for analyzing results from all YOLO training runs.
Compares both depth-based and RGB training results.
Generates comparative tables and visualizations to identify the best performing model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from utils.training_results import load_training_results

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent  # Parent workspace folder
OUTPUT_DIR = PROJECT_ROOT / "comparison_results"

# Search for runs in both project and workspace directories
RUNS_DIRS = [
    PROJECT_ROOT / "runs" / "segment",
    WORKSPACE_ROOT / "runs" / "segment",
]


def find_all_training_runs() -> dict:
    """
    Find all YOLO training runs in runs/segment/ directories.
    Searches both project root and parent workspace.
    Automatically classifies them as 'depth' or 'rgb' based on naming.

    Returns dict mapping run_name to (run_folder, model_type) tuple.
    """
    runs = {}

    for runs_dir in RUNS_DIRS:
        if not runs_dir.exists():
            continue

        for run_folder in runs_dir.iterdir():
            if not run_folder.is_dir():
                continue

            results_file = run_folder / "results.csv"
            if not results_file.exists():
                continue

            run_name = run_folder.name

            # Skip prediction folders
            if run_name.startswith("predict"):
                continue

            # Classify run type
            if run_name.startswith("depth_"):
                model_type = "depth"
                # Extract model name: depth_<model_name>_<timestamp>
                parts = run_name.split("_")
                if len(parts) >= 3:
                    # Remove 'depth_' prefix and timestamp (last 2 parts)
                    display_name = "_".join(parts[1:-2])
                else:
                    display_name = run_name
            else:
                model_type = "rgb"
                display_name = run_name

            # Handle multiple runs of same model - keep the latest
            key = f"{model_type}_{display_name}"

            if key in runs:
                # Compare timestamps or folder names to keep latest
                existing_folder = runs[key][0]
                if run_folder.name > existing_folder.name:
                    runs[key] = (run_folder, model_type, display_name)
            else:
                runs[key] = (run_folder, model_type, display_name)

    return runs


def compare_all_models(filter_type: str = None) -> tuple:
    """
    Compare all training results.

    Args:
        filter_type: Optional filter - 'depth', 'rgb', or None for all

    Returns DataFrame with comparison metrics and full results list.
    """
    print("\n" + "=" * 90)
    print("YOLO MODEL COMPARISON - ALL TRAINING RUNS")
    print("=" * 90)

    # Find all training runs
    all_runs = find_all_training_runs()

    if not all_runs:
        print("No training runs found!")
        print("Searched in:")
        for runs_dir in RUNS_DIRS:
            print(f"  - {runs_dir}")
        return None

    # Filter by type if specified
    if filter_type:
        all_runs = {k: v for k, v in all_runs.items() if v[1] == filter_type}

    # Count by type
    depth_count = sum(1 for v in all_runs.values() if v[1] == "depth")
    rgb_count = sum(1 for v in all_runs.values() if v[1] == "rgb")

    print(f"\nFound {len(all_runs)} training runs:")
    print(f"  - Depth models: {depth_count}")
    print(f"  - RGB models: {rgb_count}")
    print()

    for key, (path, model_type, display_name) in sorted(all_runs.items()):
        icon = "🎨" if model_type == "depth" else "📷"
        print(f"  {icon} {display_name}: {path.name}")

    # Load results for each model
    results = []

    for key, (run_folder, model_type, display_name) in sorted(all_runs.items()):
        metrics = load_training_results(run_folder)
        if metrics:
            metrics["model_name"] = display_name
            metrics["model_type"] = model_type
            results.append(metrics)

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
            f"{medal} {model_type_icon} {row['Model']:30} | "
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
        best_rgb = rgb_models.iloc[0]

        print(f"\n📈 DEPTH vs RGB COMPARISON:")
        print(
            f"   Best Depth Model: {best_depth['Model']:25} (mAP50-95: {best_depth['mAP50-95']:.4f})"
        )
        print(
            f"   Best RGB Model:   {best_rgb['Model']:25} (mAP50-95: {best_rgb['mAP50-95']:.4f})"
        )

        if best_rgb["mAP50-95"] > 0:
            improvement = (
                (best_depth["mAP50-95"] - best_rgb["mAP50-95"]) / best_rgb["mAP50-95"]
            ) * 100
            if improvement > 0:
                print(f"   Depth improvement over RGB: +{improvement:.2f}%")
            else:
                print(f"   RGB advantage over Depth: +{-improvement:.2f}%")

        # Average comparison
        avg_depth = depth_models["mAP50-95"].mean()
        avg_rgb = rgb_models["mAP50-95"].mean()
        print(f"\n   Average Depth mAP50-95: {avg_depth:.4f}")
        print(f"   Average RGB mAP50-95:   {avg_rgb:.4f}")


def plot_comparison(df: pd.DataFrame, results: list, output_dir: Path):
    """Generate comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except:
        plt.style.use("seaborn-darkgrid")

    # Color mapping
    def get_color(model_type):
        return "#2ecc71" if model_type == "depth" else "#3498db"

    # 1. Bar chart of mAP50-95
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.5)))

    colors = [get_color(t) for t in df["Type"]]
    bars = ax.barh(
        df["Model"], df["mAP50-95"], color=colors, edgecolor="white", linewidth=0.5
    )

    ax.set_xlabel("mAP50-95", fontsize=12)
    ax.set_title("All Models Comparison - mAP50-95", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, df["mAP50-95"]):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Depth Models"),
        Patch(facecolor="#3498db", label="RGB Models"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_mAP50-95.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, max(10, len(df) * 0.4)))
    metrics = ["mAP50-95", "mAP50", "Precision", "Recall"]

    for ax, metric in zip(axes.flatten(), metrics):
        colors = [get_color(t) for t in df["Type"]]
        bars = ax.barh(df["Model"], df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.invert_yaxis()

        for bar, val in zip(bars, df[metric]):
            if not np.isnan(val):
                ax.text(
                    val + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=8,
                )

    plt.suptitle("All Models Comparison - All Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Training curves comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use different line styles for depth vs rgb
    depth_styles = ["-", "--", "-.", ":"]
    rgb_styles = ["-", "--", "-.", ":"]
    depth_idx = 0
    rgb_idx = 0

    for r in results:
        if "results_df" in r:
            df_train = r["results_df"]
            # Find mAP50-95 column
            map_col = [c for c in df_train.columns if "mAP50-95" in c]
            if map_col:
                if r["model_type"] == "depth":
                    style = depth_styles[depth_idx % len(depth_styles)]
                    color = plt.cm.Greens(
                        0.4
                        + 0.5
                        * depth_idx
                        / max(1, sum(1 for x in results if x["model_type"] == "depth"))
                    )
                    depth_idx += 1
                else:
                    style = rgb_styles[rgb_idx % len(rgb_styles)]
                    color = plt.cm.Blues(
                        0.4
                        + 0.5
                        * rgb_idx
                        / max(1, sum(1 for x in results if x["model_type"] == "rgb"))
                    )
                    rgb_idx += 1

                label = (
                    f"[{'D' if r['model_type'] == 'depth' else 'R'}] {r['model_name']}"
                )
                ax.plot(
                    df_train["epoch"],
                    df_train[map_col[0]],
                    label=label,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle=style,
                    color=color,
                )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP50-95", fontsize=12)
    ax.set_title(
        "Training Curves - mAP50-95 Over Epochs", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Box plot comparison by type
    if len(df["Type"].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        depth_data = df[df["Type"] == "depth"]["mAP50-95"].values
        rgb_data = df[df["Type"] == "rgb"]["mAP50-95"].values

        data = []
        labels = []
        if len(depth_data) > 0:
            data.append(depth_data)
            labels.append(f"Depth (n={len(depth_data)})")
        if len(rgb_data) > 0:
            data.append(rgb_data)
            labels.append(f"RGB (n={len(rgb_data)})")

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors_box = ["#2ecc71", "#3498db"][: len(data)]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("mAP50-95", fontsize=12)
        ax.set_title(
            "mAP50-95 Distribution by Model Type", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(output_dir / "comparison_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n📊 Visualizations saved to: {output_dir}")


def export_results(df: pd.DataFrame, output_dir: Path):
    """Export comparison results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"comparison_all_models_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Results exported to: {csv_path}")


def main(filter_type: str = None, export: bool = True, plot: bool = True):
    """
    Main comparison function.

    Args:
        filter_type: Filter by model type ('depth', 'rgb', or None for all)
        export: Export results to CSV
        plot: Generate comparison plots
    """
    result = compare_all_models(filter_type)

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
    parser = argparse.ArgumentParser(
        description="Compare all YOLO training results (depth and RGB)"
    )
    parser.add_argument(
        "--type",
        choices=["depth", "rgb"],
        default=None,
        help="Filter by model type (default: show all)",
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Don't export results to CSV"
    )
    parser.add_argument("--no-plot", action="store_true", help="Don't generate plots")

    args = parser.parse_args()

    main(filter_type=args.type, export=not args.no_export, plot=not args.no_plot)
