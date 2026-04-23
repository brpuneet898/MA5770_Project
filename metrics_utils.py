"""Reusable metric utilities from the MA5770 experiments."""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def purity_score(y_true, y_pred) -> float:
    """Purity = sum over clusters of majority class count divided by N."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return float(total / n)


def safe_silhouette(X, labels, metric: str = "euclidean") -> float:
    """Silhouette score with guards for degenerate one-cluster outputs."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, metric=metric))


def clustering_report(y_true, y_pred, X=None) -> Dict[str, float]:
    """Return ARI, NMI, purity, number of clusters, and optional silhouette."""
    report = {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "Purity": purity_score(y_true, y_pred),
        "n_clusters": int(len(np.unique(y_pred))),
    }
    if X is not None:
        report["Silhouette"] = safe_silhouette(X, y_pred)
    return report
