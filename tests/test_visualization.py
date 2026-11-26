"""Unit tests for interactive HTML visual atlas builders and web explorer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fish_clustering.visualization import (
    build_atlas_dataframe,
    export_diagnostic_atlas_html,
    export_interactive_explorer,
    export_visual_atlas_html,
)


def test_build_and_export_html_atlases(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(6)],
            "dataset": ["ozfish", "ozfish", "ozfish", "fishwio", "fishwio", "schufi"],
            "source_type": ["wild"] * 5 + ["controlled"],
            "label_raw": ["sp_a", "sp_a", "sp_b", "sp_a", "sp_b", "sp_c"],
            "species": ["Caranx sp", "Caranx sp", "Chromis sp", "Caranx sp", "Chromis sp", ""],
            "video_id": ["v1"] * 6,
            "image_path": [f"p_{i}.png" for i in range(6)],
        }
    )
    coords = np.array(
        [[0.0, 0.1], [0.2, 0.0], [1.0, 1.2], [0.1, 0.2], [1.1, 1.0], [3.0, 3.0]],
        dtype=np.float32,
    )
    clusters = {"kmeans": np.array([0, 0, 1, 0, 1, -1], dtype=np.int64)}
    distortion = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.8], dtype=np.float32)
    ood = np.array([0.05, 0.06, 0.08, 0.07, 0.09, 0.75], dtype=np.float32)

    atlas_df = build_atlas_dataframe(
        manifest,
        coords,
        cluster_labels=clusters,
        distortion=distortion,
        ood_scores=ood,
    )
    assert {"x", "y", "cluster_kmeans", "distortion", "ood_score"}.issubset(atlas_df.columns)
    assert atlas_df["cluster_kmeans"].iloc[-1] == "noise (-1)"

    atlas_html = export_visual_atlas_html(
        atlas_df,
        tmp_path / "atlas.html",
        color_by="cluster_kmeans",
        title="Test Visual Atlas",
    )
    assert atlas_html.is_file()
    atlas_text = atlas_html.read_text(encoding="utf-8")
    assert "Test Visual Atlas" in atlas_text
    assert "plotly" in atlas_text.lower()

    diag_html = export_diagnostic_atlas_html(
        atlas_df,
        tmp_path / "diag.html",
        color_by="dataset",
        title="Test Diagnostic Atlas",
    )
    assert diag_html.is_file()
    diag_text = diag_html.read_text(encoding="utf-8")
    assert "Test Diagnostic Atlas" in diag_text
    assert "Neighborhood Distortion" in diag_text


def test_export_interactive_explorer_with_thumbnails_and_masks(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "frgt").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(6):
        img_rel = f"frgt/fish_{i}.png"
        Image.new("RGB", (40, 30), (30 * i + 40, 120, 180)).save(raw_root / img_rel)
        mask_rel = ""
        if i == 0:
            mask_rel = "frgt/mask_0.png"
            mask_arr = np.zeros((30, 40), dtype=np.uint8)
            mask_arr[5:25, 8:32] = 255
            Image.fromarray(mask_arr, mode="L").save(raw_root / mask_rel)
        rows.append(
            {
                "image_id": f"frgt_{i}",
                "dataset": "frgt",
                "source_type": "wild",
                "label_raw": "cluster_1" if i < 3 else "cluster_2",
                "species": "",
                "video_id": "v1",
                "image_path": img_rel,
                "mask_path": mask_rel,
            }
        )

    manifest = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(6, 8)).astype(np.float32)
    coords = np.array(
        [[0.0, 0.0], [0.4, 0.1], [0.2, 0.5], [3.0, 3.0], [3.4, 3.1], [3.2, 3.5]],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    run_data = {
        42: {
            "umap": {
                "coords": coords,
                "distortion": np.linspace(0.0, 0.5, 6, dtype=np.float32),
                "projection_metrics": {"trustworthiness": 0.95, "continuity": 0.96},
                "clusterings": {
                    "kmeans": {
                        "labels": labels,
                        "metrics": {"n_clusters": 2, "n_noise": 0, "v_measure": 1.0, "dbcv": float("nan")},
                    }
                },
            }
        }
    }

    explorer_dir = tmp_path / "explorer"
    index_path = export_interactive_explorer(
        experiment_name="explorer_unit",
        backbone="dinov3_vitb16",
        pooling="cls",
        manifest=manifest,
        embeddings=embeddings,
        run_data=run_data,
        ood_scores=np.linspace(0.1, 0.4, 6, dtype=np.float32),
        output_dir=explorer_dir,
        raw_root=raw_root,
        top_k_neighbors=3,
        thumb_max_size=64,
    )
    assert index_path.is_file()
    assert (explorer_dir / "explorer_data.js").is_file()
    json_path = explorer_dir / "explorer_data.json"
    assert json_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment"] == "explorer_unit"
    assert len(payload["samples"]) == 6
    assert payload["samples"][0]["thumb"].startswith("thumbs/")
    assert (explorer_dir / payload["samples"][0]["thumb"]).is_file()
    assert payload["samples"][0]["mask_thumb"].startswith("thumbs/")
    assert (explorer_dir / payload["samples"][0]["mask_thumb"]).is_file()
    assert len(payload["samples"][0]["neighbors"]) == 3

    geom = payload["runs"]["42"]["umap"]["clusterings"]["kmeans"]["geometry"]
    assert set(geom["centroids"].keys()) == {"0", "1"}
    assert set(geom["hulls"].keys()) == {"0", "1"}
    assert payload["runs"]["42"]["umap"]["clusterings"]["kmeans"]["metrics"]["dbcv"] is None
