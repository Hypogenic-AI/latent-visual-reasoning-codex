import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from token_dynamics import (
    DynamicsConfig,
    collapse_detected,
    generate_synthetic_data,
    run_dynamics,
    set_seed,
    summarize_config,
)


def evaluate_run(
    config: DynamicsConfig,
    seed: int,
    n_clusters: int,
    points_per_cluster: int,
    dim: int,
    cluster_std: float,
) -> Dict[str, float]:
    x, y, _ = generate_synthetic_data(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        dim=dim,
        cluster_std=cluster_std,
        seed=seed,
    )
    result = run_dynamics(x, config, seed=seed + 1234)
    ari = adjusted_rand_score(y, result.assignments)
    nmi = normalized_mutual_info_score(y, result.assignments)
    collapsed = collapse_detected(result.assignments, config.n_tokens, n_clusters)

    record = {
        "seed": seed,
        "n_clusters": n_clusters,
        "points_per_cluster": points_per_cluster,
        "dim": dim,
        "cluster_std": cluster_std,
        "ari": ari,
        "nmi": nmi,
        "sse": result.sse,
        "steps": result.steps,
        "converged": result.converged,
        "min_inter_token_dist": result.min_inter_token_dist,
        "collapsed": float(collapsed),
    }
    record.update(summarize_config(config))
    return record


def run_suite(output_dir: Path) -> pd.DataFrame:
    set_seed(42)

    configs = {
        "baseline": DynamicsConfig(n_tokens=4, n_steps=30, temperature=1.0, lr=1.0),
        "inertia": DynamicsConfig(n_tokens=4, n_steps=30, temperature=1.0, lr=1.0, momentum=0.5),
        "repulsion": DynamicsConfig(n_tokens=4, n_steps=30, temperature=1.0, lr=1.0, repulsion=0.05),
        "inertia_repulsion": DynamicsConfig(
            n_tokens=4, n_steps=30, temperature=1.0, lr=1.0, momentum=0.5, repulsion=0.05
        ),
    }

    seeds = list(range(30))
    noise_levels = [0.3, 0.6, 0.9]
    cluster_counts = [3, 4, 5]

    records: List[Dict[str, float]] = []
    for noise in noise_levels:
        for n_clusters in cluster_counts:
            for name, config in configs.items():
                for seed in seeds:
                    run_record = evaluate_run(
                        config=config,
                        seed=seed,
                        n_clusters=n_clusters,
                        points_per_cluster=80,
                        dim=6,
                        cluster_std=noise,
                    )
                    run_record["method"] = name
                    records.append(run_record)

    df = pd.DataFrame.from_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "token_dynamics_runs.csv", index=False)

    summary = (
        df.groupby(["method", "cluster_std", "n_clusters"])
        .agg({
            "ari": ["mean", "std"],
            "nmi": ["mean", "std"],
            "sse": ["mean", "std"],
            "steps": ["mean", "std"],
            "collapsed": ["mean"],
        })
        .reset_index()
    )
    summary.columns = [
        "method",
        "cluster_std",
        "n_clusters",
        "ari_mean",
        "ari_std",
        "nmi_mean",
        "nmi_std",
        "sse_mean",
        "sse_std",
        "steps_mean",
        "steps_std",
        "collapsed_rate",
    ]
    summary.to_csv(output_dir / "token_dynamics_summary.csv", index=False)

    meta = {
        "seeds": seeds,
        "noise_levels": noise_levels,
        "cluster_counts": cluster_counts,
        "configs": {name: asdict(cfg) for name, cfg in configs.items()},
    }
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return df


if __name__ == "__main__":
    run_suite(Path("results"))
