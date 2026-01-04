import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class DynamicsConfig:
    n_tokens: int
    n_steps: int = 25
    temperature: float = 1.0
    lr: float = 1.0
    momentum: float = 0.0
    repulsion: float = 0.0
    repulsion_eps: float = 1e-3
    convergence_tol: float = 1e-3


@dataclass
class DynamicsResult:
    tokens: np.ndarray
    assignments: np.ndarray
    steps: int
    converged: bool
    sse: float
    min_inter_token_dist: float


def generate_synthetic_data(
    n_clusters: int,
    points_per_cluster: int,
    dim: int,
    cluster_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate clustered Gaussian data with ground-truth labels and centers."""
    set_seed(seed)
    centers = np.random.uniform(-5.0, 5.0, size=(n_clusters, dim))
    samples = []
    labels = []
    for idx in range(n_clusters):
        cluster = centers[idx] + np.random.normal(0.0, cluster_std, size=(points_per_cluster, dim))
        samples.append(cluster)
        labels.append(np.full(points_per_cluster, idx))
    x = np.vstack(samples)
    y = np.concatenate(labels)
    return x, y, centers



def initialize_tokens(x: np.ndarray, n_tokens: int, seed: int) -> np.ndarray:
    """Initialize tokens by sampling data points with small noise."""
    set_seed(seed)
    indices = np.random.choice(x.shape[0], size=n_tokens, replace=False)
    tokens = x[indices] + np.random.normal(scale=0.05, size=(n_tokens, x.shape[1]))
    return tokens



def soft_assignments(x: np.ndarray, tokens: np.ndarray, temperature: float) -> np.ndarray:
    """Compute soft assignments using squared Euclidean distances."""
    distances = np.sum((x[:, None, :] - tokens[None, :, :]) ** 2, axis=-1)
    logits = -distances / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights



def compute_repulsion(tokens: np.ndarray, eps: float) -> np.ndarray:
    """Repulsion term to discourage token collapse (Coulomb-like)."""
    n_tokens, dim = tokens.shape
    repulsion = np.zeros_like(tokens)
    for i in range(n_tokens):
        diff = tokens[i] - tokens
        dist = np.linalg.norm(diff, axis=1) + eps
        inv_cube = 1.0 / (dist**3)
        inv_cube[i] = 0.0
        repulsion[i] = np.sum(diff * inv_cube[:, None], axis=0)
    return repulsion



def run_dynamics(x: np.ndarray, config: DynamicsConfig, seed: int) -> DynamicsResult:
    """Run token dynamics with optional momentum and repulsion."""
    tokens = initialize_tokens(x, config.n_tokens, seed=seed)
    prev_tokens = tokens.copy()

    for step in range(config.n_steps):
        weights = soft_assignments(x, tokens, config.temperature)
        weighted_sum = weights.T @ x
        weight_mass = np.sum(weights, axis=0)[:, None] + 1e-9
        mean_tokens = weighted_sum / weight_mass

        delta = config.lr * (mean_tokens - tokens)
        if config.repulsion > 0.0:
            delta += config.repulsion * compute_repulsion(tokens, config.repulsion_eps)
        if config.momentum > 0.0:
            delta += config.momentum * (tokens - prev_tokens)

        new_tokens = tokens + delta
        max_delta = np.max(np.linalg.norm(new_tokens - tokens, axis=1))
        prev_tokens = tokens
        tokens = new_tokens

        if max_delta < config.convergence_tol:
            steps = step + 1
            break
    else:
        steps = config.n_steps

    weights = soft_assignments(x, tokens, config.temperature)
    assignments = np.argmax(weights, axis=1)
    sse = float(np.sum((x - tokens[assignments]) ** 2))

    if config.n_tokens > 1:
        pairwise = np.linalg.norm(tokens[None, :, :] - tokens[:, None, :], axis=-1)
        min_inter = float(np.min(pairwise[np.nonzero(pairwise)]))
    else:
        min_inter = math.inf

    converged = steps < config.n_steps
    return DynamicsResult(
        tokens=tokens,
        assignments=assignments,
        steps=steps,
        converged=converged,
        sse=sse,
        min_inter_token_dist=min_inter,
    )


def collapse_detected(assignments: np.ndarray, n_tokens: int, n_clusters: int) -> bool:
    """Heuristic: collapse if fewer than half tokens claim distinct clusters."""
    unique_tokens = len(set(assignments.tolist()))
    return unique_tokens < max(2, int(0.5 * min(n_tokens, n_clusters)))


def summarize_config(config: DynamicsConfig) -> Dict[str, float]:
    return {
        "n_tokens": config.n_tokens,
        "n_steps": config.n_steps,
        "temperature": config.temperature,
        "lr": config.lr,
        "momentum": config.momentum,
        "repulsion": config.repulsion,
    }
