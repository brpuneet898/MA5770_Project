"""
Reusable Random-Hyperplane LSH utilities for MA5770 experiments.

The core functions here are the separated versions of the repeated LSH code used
across the text, speech/audio, CIFAR-10, and Fashion-MNIST experiments:
random_hyperplane_hash, bits_to_int, lsh_candidate_buckets,
build_candidate_pairs, connected_components, and cosine refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from collections import defaultdict
from itertools import combinations


Edge = Tuple[int, int]
BandTable = Dict[int, List[int]]


@dataclass(frozen=True)
class LSHPipelineResult:
    bits: np.ndarray
    tables: List[BandTable]
    candidate_pairs: Set[Edge]
    edges: List[Edge]
    labels: np.ndarray
    hyperplanes: np.ndarray
    total_time_sec: float
    hash_time_sec: float
    refine_time_sec: float


def make_random_hyperplanes(n_bits: int, dim: int, random_state: int = 42, dtype=np.float32) -> np.ndarray:
    """Create normalized random hyperplanes with a fixed seed."""
    rng = np.random.default_rng(random_state)
    hyperplanes = rng.normal(size=(n_bits, dim)).astype(dtype)
    hyperplanes /= np.linalg.norm(hyperplanes, axis=1, keepdims=True) + 1e-12
    return hyperplanes


def random_hyperplane_hash(X: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    """
    SimHash / random-hyperplane hash.

    X should be dense L2-normalized vectors of shape (n_samples, dim).
    hyperplanes has shape (n_bits, dim). Returns boolean signatures (n, n_bits).
    """
    proj = np.asarray(X) @ np.asarray(hyperplanes).T
    return proj >= 0.0


def bits_to_int(bit_row: Sequence[bool]) -> int:
    """Convert one boolean bit row to an integer band key."""
    out = 0
    for b in bit_row:
        out = (out << 1) | int(b)
    return out


def lsh_candidate_buckets(bits: np.ndarray, band_size: int) -> List[BandTable]:
    """
    Split bit signatures into equal-size bands and bucket identical band patterns.
    """
    n, n_bits = bits.shape
    if n_bits % band_size != 0:
        raise ValueError(f"n_bits={n_bits} must be divisible by band_size={band_size}")
    n_bands = n_bits // band_size

    tables: List[BandTable] = []
    for b in range(n_bands):
        start = b * band_size
        end = start + band_size
        table: BandTable = defaultdict(list)
        for i in range(n):
            key = bits_to_int(bits[i, start:end])
            table[key].append(i)
        tables.append(dict(table))
    return tables


def build_candidate_pairs(tables: Sequence[BandTable], max_bucket_size: int = 200) -> Set[Edge]:
    """
    Generate unique candidate pairs from LSH buckets.

    Large buckets are skipped to avoid exploding comparisons, exactly following
    the experimental design.
    """
    cand: Set[Edge] = set()
    for table in tables:
        for idxs in table.values():
            if len(idxs) < 2:
                continue
            if len(idxs) > max_bucket_size:
                continue
            for i, j in combinations(idxs, 2):
                cand.add((min(i, j), max(i, j)))
    return cand


def refine_candidate_pairs(X: np.ndarray, candidate_pairs: Iterable[Edge], cosine_threshold: float) -> List[Edge]:
    """Keep only candidate pairs whose dot/cosine similarity exceeds threshold."""
    X = np.asarray(X)
    edges: List[Edge] = []
    for i, j in candidate_pairs:
        sim = float(np.dot(X[i], X[j]))
        if sim >= cosine_threshold:
            edges.append((i, j))
    return edges


def connected_components(n: int, edges: Iterable[Edge]) -> np.ndarray:
    """Union-find connected components used to convert LSH edges into cluster labels."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in edges:
        union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def run_lsh_pipeline(
    X_jl: np.ndarray,
    n_bits: int = 96,
    band_size: int = 12,
    cosine_threshold: float = 0.65,
    max_bucket_size: int = 250,
    random_state: int = 42,
) -> LSHPipelineResult:
    """
    Full reproducible JL+LSH graph clustering pipeline.

    X_jl must be dense and L2-normalized. The default parameters reproduce the
    common setup in the text experiments; cosine_threshold can be adjusted by
    modality, e.g. 0.70/0.72 for speech and 0.80 for images.
    """
    X_jl = np.asarray(X_jl)
    t0 = perf_counter()
    hyperplanes = make_random_hyperplanes(n_bits=n_bits, dim=X_jl.shape[1], random_state=random_state)
    bits = random_hyperplane_hash(X_jl, hyperplanes)
    tables = lsh_candidate_buckets(bits, band_size=band_size)
    candidate_pairs = build_candidate_pairs(tables, max_bucket_size=max_bucket_size)
    mid = perf_counter()
    edges = refine_candidate_pairs(X_jl, candidate_pairs, cosine_threshold=cosine_threshold)
    labels = connected_components(n=X_jl.shape[0], edges=edges)
    total = perf_counter() - t0
    hash_time = mid - t0
    return LSHPipelineResult(
        bits=bits,
        tables=tables,
        candidate_pairs=candidate_pairs,
        edges=edges,
        labels=labels,
        hyperplanes=hyperplanes,
        total_time_sec=total,
        hash_time_sec=hash_time,
        refine_time_sec=total - hash_time,
    )
