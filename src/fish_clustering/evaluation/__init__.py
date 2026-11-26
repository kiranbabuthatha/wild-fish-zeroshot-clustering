"""Clustering, projection quality (Trustworthiness/Continuity), and RQ2 metrics."""

from fish_clustering.evaluation.clustering_metrics import (
    compute_label_free_metrics,
    compute_supervised_clustering_metrics,
    evaluate_clustering_results,
)
from fish_clustering.evaluation.projection_metrics import (
    compute_continuity,
    compute_pointwise_distortion,
    compute_projection_quality,
    compute_trustworthiness,
)
from fish_clustering.evaluation.rq2_metrics import (
    compute_cross_dataset_species_match,
    compute_dataset_mixing_entropy,
    compute_knn_ood_scores,
    compute_self_knn_ood_scores,
)

__all__ = [
    "compute_continuity",
    "compute_cross_dataset_species_match",
    "compute_dataset_mixing_entropy",
    "compute_knn_ood_scores",
    "compute_label_free_metrics",
    "compute_pointwise_distortion",
    "compute_projection_quality",
    "compute_self_knn_ood_scores",
    "compute_supervised_clustering_metrics",
    "compute_trustworthiness",
    "evaluate_clustering_results",
]
