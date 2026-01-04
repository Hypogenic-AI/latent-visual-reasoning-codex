import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-9))


def paired_tests(df: pd.DataFrame, output_path: Path) -> None:
    comparisons = [
        ("baseline", "inertia"),
        ("baseline", "repulsion"),
        ("baseline", "inertia_repulsion"),
    ]
    results = []

    grouped = df.groupby(["cluster_std", "n_clusters"])
    for (noise, n_clusters), group in grouped:
        for a, b in comparisons:
            a_vals = group[group["method"] == a].sort_values("seed")["ari"].to_numpy()
            b_vals = group[group["method"] == b].sort_values("seed")["ari"].to_numpy()
            if len(a_vals) != len(b_vals):
                continue
            t_stat, p_val = stats.ttest_rel(b_vals, a_vals)
            effect = cohens_d(b_vals, a_vals)
            ci_low, ci_high = stats.t.interval(
                0.95, len(b_vals) - 1, loc=(b_vals - a_vals).mean(), scale=stats.sem(b_vals - a_vals)
            )
            results.append(
                {
                    "noise": float(noise),
                    "n_clusters": int(n_clusters),
                    "comparison": f"{b}_vs_{a}",
                    "mean_diff": float((b_vals - a_vals).mean()),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "cohens_d": effect,
                }
            )

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def plot_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["ari_mean", "steps_mean", "collapsed_rate"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in summary_df["method"].unique():
            subset = summary_df[summary_df["method"] == method]
            for n_clusters in sorted(summary_df["n_clusters"].unique()):
                data = subset[subset["n_clusters"] == n_clusters]
                ax.plot(
                    data["cluster_std"],
                    data[metric],
                    marker="o",
                    label=f"{method} (K={n_clusters})",
                )
        ax.set_xlabel("Cluster noise (std)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{metric} across noise levels")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    results_dir = Path("results")
    runs_path = results_dir / "token_dynamics_runs.csv"
    summary_path = results_dir / "token_dynamics_summary.csv"

    df = pd.read_csv(runs_path)
    summary_df = pd.read_csv(summary_path)

    paired_tests(df, results_dir / "stat_tests.json")
    plot_summary(summary_df, results_dir / "plots")


if __name__ == "__main__":
    main()
