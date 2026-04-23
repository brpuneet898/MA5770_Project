"""
Reusable Johnson--Lindenstrauss (JL) projection utilities for MA5770 experiments.

These functions were separated from the experiment notebooks/scripts so the same
projection, normalization, and distortion-validation steps can be reused across
text, audio, and image experiments with fixed random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple, Dict

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection


ArrayLike = np.ndarray


@dataclass(frozen=True)
class JLProjectionResult:
    """Container returned by fit_jl_projection."""

    X_jl: np.ndarray
    transformer: SparseRandomProjection
    elapsed_sec: float


def l2_normalize(X, axis: int = 1):
    """L2-normalize a dense or sparse matrix exactly as used in the experiments."""
    return normalize(X, norm="l2", axis=axis)


def fit_jl_projection(
    X,
    n_components: int = 512,
    random_state: int = 42,
    dense_output: bool = True,
    normalize_output: bool = True,
) -> JLProjectionResult:
    """
    Fit and apply SparseRandomProjection, then optionally L2-normalize output.

    Parameters
    ----------
    X:
        Dense or sparse input feature matrix of shape (n_samples, n_features).
    n_components:
        Target JL dimension. Experiments used 512 for text/CIFAR/Speech Commands,
        256 for Spoken Digit, and 128 for Fashion-MNIST.
    random_state:
        Fixed seed for reproducibility.
    dense_output:
        Whether to return a dense projected matrix.
    normalize_output:
        Whether to L2-normalize projected vectors.
    """
    rp = SparseRandomProjection(
        n_components=n_components,
        dense_output=dense_output,
        random_state=random_state,
    )
    t0 = perf_counter()
    X_jl = rp.fit_transform(X)
    elapsed = perf_counter() - t0
    if normalize_output:
        X_jl = l2_normalize(X_jl, axis=1)
    if sparse.issparse(X_jl):
        X_jl = X_jl.toarray()
    return JLProjectionResult(X_jl=np.asarray(X_jl), transformer=rp, elapsed_sec=elapsed)


def apply_jl_projection(
    X,
    transformer: SparseRandomProjection,
    normalize_output: bool = True,
) -> np.ndarray:
    """Apply an already fitted JL transformer to new data."""
    X_jl = transformer.transform(X)
    if normalize_output:
        X_jl = l2_normalize(X_jl, axis=1)
    if sparse.issparse(X_jl):
        X_jl = X_jl.toarray()
    return np.asarray(X_jl)


def cosine_sim_rows(X, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine/dot similarity for L2-normalized dense or sparse matrices."""
    if sparse.issparse(X):
        return np.array([X[i].multiply(X[j]).sum() for i, j in zip(idx_a, idx_b)])
    X = np.asarray(X)
    return np.sum(X[idx_a] * X[idx_b], axis=1)


def sample_pair_indices(n_samples: int, n_pairs: int = 3000, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate reproducible random index pairs for JL distortion checking."""
    rng = np.random.default_rng(random_state)
    idx_a = rng.integers(0, n_samples, size=n_pairs)
    idx_b = rng.integers(0, n_samples, size=n_pairs)
    return idx_a, idx_b


def jl_cosine_distortion(
    X_original,
    X_jl: np.ndarray,
    n_pairs: int = 3000,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Validate JL preservation using sampled cosine-distance distortion.

    Returns mean absolute distortion and selected quantiles. This mirrors the
    experiment cells where d = 1 - cosine_similarity was compared before/after JL.
    """
    n = X_original.shape[0]
    idx_a, idx_b = sample_pair_indices(n, n_pairs=n_pairs, random_state=random_state)

    orig_sim = cosine_sim_rows(X_original, idx_a, idx_b)
    jl_sim = cosine_sim_rows(X_jl, idx_a, idx_b)

    orig_dist = 1.0 - orig_sim
    jl_dist = 1.0 - jl_sim
    abs_distortion = np.abs(jl_dist - orig_dist)

    return {
        "mean_abs_distortion": float(abs_distortion.mean()),
        "median_abs_distortion": float(np.quantile(abs_distortion, 0.50)),
        "p95_abs_distortion": float(np.quantile(abs_distortion, 0.95)),
        "max_abs_distortion": float(abs_distortion.max()),
        "n_pairs": int(n_pairs),
    }
