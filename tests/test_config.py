"""Unit tests for YAML configuration loading and device resolution."""

from pathlib import Path

import pytest

from fish_clustering.config import ThesisConfig, load_config, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_and_smoke_configs() -> None:
    default_cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert isinstance(default_cfg, ThesisConfig)
    assert default_cfg.experiment.name == "wild_fish_default"
    assert default_cfg.experiment.seeds == [42, 43, 44]
    assert "frgt" in default_cfg.data.datasets
    assert default_cfg.model.l2_normalize is True

    smoke_cfg = load_config(REPO_ROOT / "configs" / "smoke_test.yaml")
    assert smoke_cfg.experiment.name == "frgt_smoke_test"
    assert smoke_cfg.data.datasets == ["frgt"]
    assert smoke_cfg.data.max_images_per_species == 20
    assert smoke_cfg.data.manifest_path == Path("data/manifests/smoke_frgt.csv")

    fishwio_cfg = load_config(REPO_ROOT / "configs" / "fishwio_dinov3_demo.yaml")
    assert fishwio_cfg.experiment.name == "fishwio_dinov3_demo"
    assert fishwio_cfg.data.datasets == ["fishwio"]
    assert fishwio_cfg.model.backbone == "dinov3_vitb16"


def test_load_config_validation_errors(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        """
experiment:
  name: bad_test
  seeds: []
  device: auto
data:
  manifest_path: data/manifests/test.csv
  raw_root: data/raw
  datasets: [frgt]
model:
  backbone: dinov3_vitl16
reduction:
  methods: [pca]
clustering:
  methods: [kmeans]
evaluation:
  k_neighbors: 10
visualization:
  export_html: false
cropping:
  detector: none
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="seeds"):
        load_config(bad_yaml)


def test_load_config_rejects_missing_section(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text("experiment:\n  name: incomplete\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required config section"):
        load_config(incomplete)


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")
