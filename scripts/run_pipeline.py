"""Run zero-shot feature extraction, DR, clustering, evaluation, and visual export."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fish_clustering.clustering import run_clustering_suite, run_reductions
from fish_clustering.config import ThesisConfig, load_config
from fish_clustering.data import apply_sampling_filters, build_manifest
from fish_clustering.evaluation import (
    compute_cross_dataset_species_match,
    compute_dataset_mixing_entropy,
    compute_knn_ood_scores,
    compute_projection_quality,
    compute_self_knn_ood_scores,
    evaluate_clustering_results,
)
from fish_clustering.models import (
    BaseFeatureExtractor,
    create_extractor,
    extract_or_load_embeddings,
)
from fish_clustering.visualization import (
    build_atlas_dataframe,
    export_diagnostic_atlas_html,
    export_interactive_explorer,
    export_static_figures,
    export_visual_atlas_html,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

_STRING_MANIFEST_COLS = (
    "image_id",
    "dataset",
    "source_type",
    "image_path",
    "mask_path",
    "label_raw",
    "family",
    "genus",
    "species",
    "video_id",
    "split",
)


class DummyFeatureExtractor(BaseFeatureExtractor):
    """Lightweight deterministic image-grid extractor for offline smoke checks."""

    def __init__(self, *, pooling: str = "cls", image_size: int = 32) -> None:
        super().__init__(
            "dummy_grid",
            pooling=pooling,
            device="cpu",
            image_size=image_size,
        )

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for img in images:
            resized = img.convert("RGB").resize((8, 8), resample=Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            q00 = arr[:4, :4].mean(axis=(0, 1))
            q01 = arr[:4, 4:].mean(axis=(0, 1))
            q10 = arr[4:, :4].mean(axis=(0, 1))
            q11 = arr[4:, 4:].mean(axis=(0, 1))
            global_mean = arr.mean(axis=(0, 1))
            global_std = arr.std(axis=(0, 1))
            rows.append(np.concatenate([q00, q01, q10, q11, global_mean, global_std]))
        return np.stack(rows, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running the full thesis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the zero-shot fish clustering and cross-dataset pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_test.yaml"),
        help="Path to experiment YAML configuration file.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip datasets missing on disk when auto-building the manifest.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute backbone embeddings even if a matching cache exists.",
    )
    parser.add_argument(
        "--extractor-backend",
        choices=("auto", "dummy"),
        default="auto",
        help="Use 'auto' for configured ViT/BioCLIP backbones or 'dummy' for offline smoke checks.",
    )
    return parser.parse_args()


def _load_or_build_manifest(cfg: ThesisConfig, *, skip_missing: bool) -> pd.DataFrame:
    if cfg.data.manifest_path.is_file():
        df = pd.read_csv(cfg.data.manifest_path, low_memory=False)
    else:
        LOGGER.info("Manifest %s not found; building from %s.", cfg.data.manifest_path, cfg.data.raw_root)
        df = build_manifest(cfg.data.raw_root, cfg.data.datasets, skip_missing=skip_missing)
        cfg.data.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.data.manifest_path, index=False)

    for col in _STRING_MANIFEST_COLS:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "").astype(str)
    if "is_species_level" in df.columns:
        df["is_species_level"] = df["is_species_level"].apply(
            lambda v: v is True or (isinstance(v, str) and v.strip().lower() == "true")
        )
    return df


def _select_true_labels(manifest: pd.DataFrame) -> list[str]:
    species_series = (
        manifest["species"].fillna("").astype(str).str.strip()
        if "species" in manifest.columns
        else pd.Series("", index=manifest.index)
    )
    raw_series = (
        manifest["label_raw"].fillna("").astype(str).str.strip()
        if "label_raw" in manifest.columns
        else pd.Series("", index=manifest.index)
    )
    if (species_series != "").mean() >= 0.5:
        return [sp if sp else raw for sp, raw in zip(species_series, raw_series)]
    return raw_series.tolist()


def run_experiment(
    cfg: ThesisConfig,
    *,
    extractor: BaseFeatureExtractor | None = None,
    extractor_backend: str = "auto",
    skip_missing: bool = False,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Run extraction, reduction, clustering, evaluation, and atlas export for a ThesisConfig."""
    raw_manifest = _load_or_build_manifest(cfg, skip_missing=skip_missing)
    manifest = apply_sampling_filters(
        raw_manifest,
        max_crops_per_video=cfg.data.max_crops_per_video,
        max_images_per_species=cfg.data.max_images_per_species,
        seed=cfg.experiment.seeds[0],
    )
    if manifest.empty:
        raise ValueError("Manifest contains 0 samples after applying configured sampling filters.")

    if extractor is None:
        if extractor_backend == "dummy":
            extractor = DummyFeatureExtractor(
                pooling=cfg.model.pooling,
                image_size=cfg.model.image_size,
            )
        else:
            extractor = create_extractor(
                cfg.model.backbone,
                pooling=cfg.model.pooling,
                device=cfg.experiment.device,
                image_size=cfg.model.image_size,
            )

    cached = extract_or_load_embeddings(
        manifest,
        extractor,
        cache_dir=cfg.model.cache_dir,
        raw_root=cfg.data.raw_root,
        batch_size=cfg.model.batch_size,
        l2_normalize=cfg.model.l2_normalize,
        use_foreground_mask=cfg.data.use_foreground_mask,
        equalize_size=cfg.data.equalize_resolution,
        force_recompute=force_recompute,
        show_progress=True,
    )
    embeddings = cached.embeddings
    true_labels = _select_true_labels(manifest)
    unique_true = {lbl for lbl in true_labels if lbl != ""}
    n_clusters = len(unique_true) if 2 <= len(unique_true) < len(manifest) else None

    datasets_col = manifest["dataset"].tolist() if "dataset" in manifest.columns else ["unknown"] * len(manifest)
    species_col = manifest["species"].tolist() if "species" in manifest.columns else [""] * len(manifest)
    ood_scores = compute_self_knn_ood_scores(
        embeddings,
        k_neighbors=cfg.evaluation.k_neighbors,
    )

    rq2_summary: dict[str, Any] = {}
    mixing_entropies: np.ndarray | None = None
    if cfg.evaluation.compute_rq2_mixing:
        mixing = compute_dataset_mixing_entropy(
            embeddings,
            datasets_col,
            k_neighbors=cfg.evaluation.k_neighbors,
        )
        mixing_entropies = mixing["per_sample_entropy"]
        cross_match = compute_cross_dataset_species_match(
            embeddings,
            datasets_col,
            species_col,
            k_neighbors=cfg.evaluation.k_neighbors,
        )
        rq2_summary = {
            "mean_normalized_dataset_entropy": mixing["mean_normalized_entropy"],
            "mean_same_dataset_neighbor_rate": mixing["mean_same_dataset_rate"],
            "n_datasets": mixing["n_datasets"],
            "cross_dataset_species_match": cross_match,
            "mean_self_knn_ood": float(np.mean(ood_scores)),
        }
        is_schufi = np.asarray([d == "schufi" for d in datasets_col], dtype=bool)
        if np.any(is_schufi) and np.any(~is_schufi):
            schufi_ood = compute_knn_ood_scores(
                embeddings[~is_schufi],
                embeddings[is_schufi],
                k_neighbors=cfg.evaluation.k_neighbors,
            )
            rq2_summary["mean_schufi_vs_open_ood"] = float(np.mean(schufi_ood))

    clustering_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    vis_paths: list[str] = []
    explorer_run_data: dict[int, dict[str, dict[str, Any]]] = {}

    metrics_dir = cfg.evaluation.output_dir / cfg.experiment.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = cfg.visualization.output_dir / cfg.experiment.name

    for seed in cfg.experiment.seeds:
        explorer_run_data[seed] = {}
        if cfg.clustering.run_in_high_dim:
            high_res = run_clustering_suite(
                embeddings,
                methods=cfg.clustering.methods,
                n_clusters=n_clusters,
                hdbscan_min_cluster_size=cfg.clustering.hdbscan_min_cluster_size,
                seed=seed,
                space="high_dim",
            )
            clustering_rows.extend(
                evaluate_clustering_results(
                    high_res,
                    embeddings,
                    true_labels=true_labels,
                    compute_label_free=cfg.evaluation.compute_label_free,
                    metric="cosine" if cfg.model.l2_normalize else "euclidean",
                )
            )

        reductions = run_reductions(
            embeddings,
            methods=cfg.reduction.methods,
            n_components=cfg.reduction.n_components,
            tsne_perplexity=cfg.reduction.tsne_perplexity,
            umap_n_neighbors=cfg.reduction.umap_n_neighbors,
            umap_min_dist=cfg.reduction.umap_min_dist,
            seed=seed,
        )

        for red_name, red_res in reductions.items():
            distortion: np.ndarray | None = None
            pq_summary: dict[str, Any] = {}
            if cfg.evaluation.compute_projection_quality and len(manifest) >= 5:
                pq = compute_projection_quality(
                    embeddings,
                    red_res.coords,
                    n_neighbors=cfg.evaluation.k_neighbors,
                )
                distortion = pq["per_point_distortion"]
                pq_summary = {
                    "seed": seed,
                    "reduction": red_name,
                    "trustworthiness": pq["trustworthiness"],
                    "continuity": pq["continuity"],
                    "mean_neighborhood_jaccard": pq["mean_neighborhood_jaccard"],
                    "k_neighbors": pq["k_neighbors"],
                }
                projection_rows.append(pq_summary)

            low_res = run_clustering_suite(
                red_res.coords,
                methods=cfg.clustering.methods,
                n_clusters=n_clusters,
                hdbscan_min_cluster_size=cfg.clustering.hdbscan_min_cluster_size,
                seed=seed,
                space=red_name,
            )
            low_eval_rows = evaluate_clustering_results(
                low_res,
                red_res.coords,
                true_labels=true_labels,
                compute_label_free=cfg.evaluation.compute_label_free,
                metric="euclidean",
            )
            clustering_rows.extend(low_eval_rows)

            eval_by_method = {row["method"]: row for row in low_eval_rows}
            explorer_run_data[seed][red_name] = {
                "coords": red_res.coords,
                "distortion": distortion,
                "projection_metrics": pq_summary,
                "clusterings": {
                    method: {
                        "labels": res.labels,
                        "metrics": eval_by_method.get(method, {}),
                    }
                    for method, res in low_res.items()
                },
            }

            if cfg.visualization.export_html:
                atlas_df = build_atlas_dataframe(
                    manifest,
                    red_res.coords,
                    cluster_labels={m: r.labels for m, r in low_res.items()},
                    distortion=distortion,
                    ood_scores=ood_scores,
                )
                color_col = "species" if (manifest["species"] != "").mean() >= 0.5 else "label_raw"
                atlas_path = export_visual_atlas_html(
                    atlas_df,
                    vis_dir / f"{red_name}_seed{seed}.html",
                    color_by=color_col,
                    title=f"{cfg.experiment.name} - {red_name.upper()} (seed {seed})",
                )
                diag_path = export_diagnostic_atlas_html(
                    atlas_df,
                    vis_dir / f"{red_name}_seed{seed}_diagnostics.html",
                    color_by=color_col,
                    title=f"{cfg.experiment.name} - {red_name.upper()} Diagnostics (seed {seed})",
                )
                vis_paths.extend([str(atlas_path), str(diag_path)])

    explorer_html_path: Path | None = None
    if cfg.visualization.export_html:
        explorer_html_path = export_interactive_explorer(
            experiment_name=cfg.experiment.name,
            backbone=extractor.backbone_name,
            pooling=extractor.pooling,
            manifest=manifest,
            embeddings=embeddings,
            run_data=explorer_run_data,
            ood_scores=ood_scores,
            output_dir=vis_dir / "explorer",
            raw_root=cfg.data.raw_root,
            rq2_summary=rq2_summary,
        )

    static_figure_paths = export_static_figures(
        experiment_name=cfg.experiment.name,
        manifest=manifest,
        embeddings=embeddings,
        run_data=explorer_run_data,
        projection_rows=projection_rows,
        ood_scores=ood_scores,
        output_dir=vis_dir / "figures",
        raw_root=cfg.data.raw_root,
        mixing_entropies=mixing_entropies,
        true_labels=true_labels,
        primary_seed=cfg.experiment.seeds[0],
    )

    clustering_csv = metrics_dir / "clustering_metrics.csv"
    pd.DataFrame(clustering_rows).to_csv(clustering_csv, index=False)

    projection_csv = metrics_dir / "projection_metrics.csv"
    if projection_rows:
        pd.DataFrame(projection_rows).to_csv(projection_csv, index=False)

    summary: dict[str, Any] = {
        "experiment": cfg.experiment.name,
        "backbone": extractor.backbone_name,
        "pooling": extractor.pooling,
        "num_samples": int(len(manifest)),
        "embedding_dim": int(embeddings.shape[1]),
        "cache_path": str(cached.cache_path) if cached.cache_path else None,
        "clustering_metrics_path": str(clustering_csv),
        "projection_metrics_path": str(projection_csv) if projection_rows else None,
        "explorer_html_path": str(explorer_html_path) if explorer_html_path else None,
        "static_figure_paths": static_figure_paths,
        "visual_atlas_paths": vis_paths,
        "rq2": rq2_summary,
    }
    summary_json = metrics_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info(
        "Completed %s (%d samples): saved metrics to %s",
        cfg.experiment.name,
        len(manifest),
        metrics_dir,
    )
    return summary


def main() -> None:
    """Entry point for the end-to-end clustering and evaluation pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    run_experiment(
        cfg,
        extractor_backend=args.extractor_backend,
        skip_missing=args.skip_missing,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
