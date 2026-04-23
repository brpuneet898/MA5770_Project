"""
Small offline reproducibility demo for the separated MA5770 JL + LSH functions.

Run:
    python reproducibility_demo.py

It uses synthetic vectors only, so it does not download any dataset. The fixed
seeds make the candidate count, edge count, labels, and JL distortion reproducible.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize

from jl_projection import fit_jl_projection, jl_cosine_distortion
from lsh_core import run_lsh_pipeline
from metrics_utils import clustering_report


def make_synthetic_near_duplicate_data(n_base: int = 120, dim: int = 800, random_state: int = 7):
    rng = np.random.default_rng(random_state)
    base = rng.normal(size=(n_base, dim)).astype(np.float32)
    base = normalize(base, norm="l2", axis=1)

    # Add one noisy near-duplicate for each base vector.
    dup = base + 0.05 * rng.normal(size=base.shape).astype(np.float32)
    dup = normalize(dup, norm="l2", axis=1)

    X = np.vstack([base, dup])
    y_true = np.arange(n_base).repeat(2)
    order = rng.permutation(X.shape[0])
    return X[order], y_true[order]


def main():
    X, y_true = make_synthetic_near_duplicate_data()

    jl = fit_jl_projection(X, n_components=128, random_state=42)
    distortion = jl_cosine_distortion(X, jl.X_jl, n_pairs=1000, random_state=0)

    lsh = run_lsh_pipeline(
        jl.X_jl,
        n_bits=96,
        band_size=12,
        cosine_threshold=0.72,
        max_bucket_size=250,
        random_state=42,
    )

    report = clustering_report(y_true, lsh.labels, X=jl.X_jl)

    print("JL shape:", jl.X_jl.shape)
    print("JL distortion:", distortion)
    print("LSH candidates:", len(lsh.candidate_pairs))
    print("LSH edges kept:", len(lsh.edges))
    print("LSH clusters:", len(np.unique(lsh.labels)))
    print("Report:", report)


if __name__ == "__main__":
    main()
