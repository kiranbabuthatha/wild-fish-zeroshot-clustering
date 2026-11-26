"""End-to-end integration test for scripts/run_pipeline.py."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from fish_clustering.config import load_config
from scripts.run_pipeline import DummyFeatureExtractor, run_experiment


def _write_colored_crop(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def test_run_experiment_end_to_end(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    oz_crops = raw_root / "ozfish" / "crops"
    (raw_root / "ozfish" / "metadata").mkdir(parents=True, exist_ok=True)
    oz_rows = ["uid,file_name,family,genus,species"]

    fw_root = raw_root / "fishwio" / "extracted" / "data_paper_dataset"

    for i in range(6):
        fname_a = f"A0001_L.avi.{i}.png"
        _write_colored_crop(oz_crops / fname_a, (220 - i * 3, 20, 20))
        oz_rows.append(f"a{i},{fname_a},Carangidae,Caranx,melampygus")

        fname_b = f"B0001_L.avi.{i}.png"
        _write_colored_crop(oz_crops / fname_b, (20, 20, 220 - i * 3))
        oz_rows.append(f"b{i},{fname_b},Pomacentridae,Chromis,chromis")

        _write_colored_crop(
            fw_root / "Caranx_melampygus" / f"FW_a_{i}.jpeg",
            (215 - i * 2, 25, 25),
        )
        _write_colored_crop(
            fw_root / "Chromis_chromis" / f"FW_b_{i}.jpeg",
            (25, 25, 215 - i * 2),
        )

    (raw_root / "ozfish" / "metadata" / "crop_metadata.csv").write_text(
        "\n".join(oz_rows) + "\n", encoding="utf-8"
    )

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(
        f"""
experiment:
  name: synthetic_e2e
  seeds: [42]
  device: cpu
data:
  manifest_path: {tmp_path / "manifests" / "all.csv"}
  raw_root: {raw_root}
  datasets: [ozfish, fishwio]
  max_crops_per_video: null
  max_images_per_species: null
  use_foreground_mask: false
  equalize_resolution: null
model:
  backbone: dinov3_vitb16
  pooling: cls
  batch_size: 8
  image_size: 32
  l2_normalize: true
  cache_dir: {tmp_path / "outputs" / "embeddings"}
reduction:
  methods: [pca, tsne, umap]
  n_components: 2
  tsne_perplexity: 5
  umap_n_neighbors: 5
  umap_min_dist: 0.1
clustering:
  methods: [hdbscan, ward, kmeans, gmm]
  hdbscan_min_cluster_size: 4
  run_in_high_dim: true
evaluation:
  compute_projection_quality: true
  k_neighbors: 5
  compute_label_free: true
  compute_rq2_mixing: true
  output_dir: {tmp_path / "outputs" / "metrics"}
visualization:
  export_html: true
  output_dir: {tmp_path / "outputs" / "visual_atlas"}
cropping:
  detector: none
  confidence_threshold: 0.5
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    summary = run_experiment(cfg, extractor=DummyFeatureExtractor())

    assert summary["num_samples"] == 24
    assert Path(summary["cache_path"]).is_file()
    assert Path(summary["clustering_metrics_path"]).is_file()
    assert Path(summary["projection_metrics_path"]).is_file()
    explorer_html = Path(summary["explorer_html_path"])
    assert explorer_html.is_file()
    assert (explorer_html.parent / "explorer_data.json").is_file()
    assert (explorer_html.parent / "explorer_data.js").is_file()
    assert any((explorer_html.parent / "thumbs").iterdir())
    assert len(summary["static_figure_paths"]) >= 8
    for fig_path in summary["static_figure_paths"]:
        assert Path(fig_path).is_file()
    assert len(summary["visual_atlas_paths"]) == 6
    for html_path in summary["visual_atlas_paths"]:
        assert Path(html_path).is_file()

    summary_json = tmp_path / "outputs" / "metrics" / "synthetic_e2e" / "summary.json"
    assert summary_json.is_file()
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded["rq2"]["cross_dataset_species_match"]["n_shared_species"] == 2

    clustering_df = pd.read_csv(summary["clustering_metrics_path"])
    assert set(clustering_df["space"]) == {"high_dim", "pca", "tsne", "umap"}
