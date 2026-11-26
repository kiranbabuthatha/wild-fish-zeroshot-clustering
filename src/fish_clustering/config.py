"""YAML configuration loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass
class ExperimentConfig:
    """Top-level experiment execution settings."""

    name: str
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "auto"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("experiment.name must be a non-empty string.")
        if not self.seeds:
            raise ValueError("experiment.seeds must contain at least one seed.")
        if any(not isinstance(s, int) or isinstance(s, bool) for s in self.seeds):
            raise ValueError("experiment.seeds must contain only integers.")
        if self.device not in ALLOWED_DEVICES:
            raise ValueError(
                f"experiment.device must be one of {sorted(ALLOWED_DEVICES)}, got {self.device!r}."
            )


@dataclass
class DataConfig:
    """Dataset manifest, filtering, and preprocessing settings."""

    manifest_path: Path
    raw_root: Path
    datasets: list[str]
    max_crops_per_video: int | None = None
    max_images_per_species: int | None = None
    min_images_per_species: int | None = None
    top_n_species: int | None = None
    use_foreground_mask: bool = False
    equalize_resolution: int | None = None

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        self.raw_root = Path(self.raw_root)
        if not self.datasets:
            raise ValueError("data.datasets must contain at least one dataset name.")
        if self.max_crops_per_video is not None and self.max_crops_per_video <= 0:
            raise ValueError("data.max_crops_per_video must be positive or null.")
        if self.max_images_per_species is not None and self.max_images_per_species <= 0:
            raise ValueError("data.max_images_per_species must be positive or null.")
        if self.min_images_per_species is not None and self.min_images_per_species <= 0:
            raise ValueError("data.min_images_per_species must be positive or null.")
        if self.top_n_species is not None and self.top_n_species <= 0:
            raise ValueError("data.top_n_species must be positive or null.")
        if isinstance(self.equalize_resolution, bool):
            raise ValueError("data.equalize_resolution must be an integer pixel size or null.")
        if self.equalize_resolution is not None and self.equalize_resolution <= 0:
            raise ValueError("data.equalize_resolution must be positive or null.")


@dataclass
class ModelConfig:
    """Frozen vision backbone feature extraction settings."""

    backbone: str
    pooling: str = "cls"
    batch_size: int = 32
    image_size: int = 224
    l2_normalize: bool = True
    cache_dir: Path = Path("outputs/embeddings")

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if not self.backbone or not self.backbone.strip():
            raise ValueError("model.backbone must be a non-empty string.")
        if self.batch_size <= 0:
            raise ValueError("model.batch_size must be positive.")
        if self.image_size <= 0:
            raise ValueError("model.image_size must be positive.")


@dataclass
class ReductionConfig:
    """Dimensionality reduction settings for PCA, t-SNE, and UMAP."""

    methods: list[str] = field(default_factory=lambda: ["pca", "tsne", "umap"])
    n_components: int = 2
    tsne_perplexity: float = 30.0
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("reduction.methods must contain at least one method.")
        if self.n_components < 2:
            raise ValueError("reduction.n_components must be at least 2.")
        if self.tsne_perplexity <= 0:
            raise ValueError("reduction.tsne_perplexity must be positive.")
        if self.umap_n_neighbors < 2:
            raise ValueError("reduction.umap_n_neighbors must be at least 2.")
        if self.umap_min_dist < 0:
            raise ValueError("reduction.umap_min_dist must be non-negative.")


@dataclass
class ClusteringConfig:
    """Clustering algorithm settings."""

    methods: list[str] = field(default_factory=lambda: ["hdbscan", "ward", "kmeans", "gmm"])
    hdbscan_min_cluster_size: int = 15
    run_in_high_dim: bool = True

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("clustering.methods must contain at least one method.")
        if self.hdbscan_min_cluster_size < 2:
            raise ValueError("clustering.hdbscan_min_cluster_size must be at least 2.")


@dataclass
class EvaluationConfig:
    """Projection quality, clustering, and mixing evaluation settings."""

    compute_projection_quality: bool = True
    k_neighbors: int = 15
    compute_label_free: bool = True
    compute_rq2_mixing: bool = True
    output_dir: Path = Path("outputs/metrics")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.k_neighbors < 1:
            raise ValueError("evaluation.k_neighbors must be at least 1.")


@dataclass
class VisualizationConfig:
    """Interactive plot export settings."""

    export_html: bool = True
    output_dir: Path = Path("outputs/plots")

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)


@dataclass
class CroppingConfig:
    """Optional detector settings for uncropped video frames."""

    detector: str = "none"
    confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("cropping.confidence_threshold must be between 0.0 and 1.0.")


@dataclass
class ThesisConfig:
    """Full typed configuration for the wild fish clustering pipeline."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    reduction: ReductionConfig
    clustering: ClusteringConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig
    cropping: CroppingConfig
    config_path: Path | None = None


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in raw:
        raise ValueError(f"Missing required config section: {name!r}.")
    section = raw[name]
    if not isinstance(section, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return section


def _build_section(cls: type[Any], raw: dict[str, Any], section_name: str) -> Any:
    section_data = _require_section(raw, section_name)
    try:
        return cls(**section_data)
    except TypeError as exc:
        raise ValueError(f"Invalid fields in config section {section_name!r}: {exc}") from exc


def resolve_device(device: str) -> str:
    """Resolve 'auto' into an available PyTorch compute device string."""
    if device not in ALLOWED_DEVICES:
        raise ValueError(
            f"Unsupported device {device!r}; expected one of {sorted(ALLOWED_DEVICES)}."
        )
    if device != "auto":
        return device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def load_config(path: str | Path) -> ThesisConfig:
    """Load and validate an experiment YAML file into a ThesisConfig instance."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a top-level YAML mapping.")

    required_sections = {
        "experiment",
        "data",
        "model",
        "reduction",
        "clustering",
        "evaluation",
        "visualization",
        "cropping",
    }
    extra_sections = set(raw.keys()) - required_sections
    if extra_sections:
        raise ValueError(f"Unknown top-level config sections: {sorted(extra_sections)}")

    return ThesisConfig(
        experiment=_build_section(ExperimentConfig, raw, "experiment"),
        data=_build_section(DataConfig, raw, "data"),
        model=_build_section(ModelConfig, raw, "model"),
        reduction=_build_section(ReductionConfig, raw, "reduction"),
        clustering=_build_section(ClusteringConfig, raw, "clustering"),
        evaluation=_build_section(EvaluationConfig, raw, "evaluation"),
        visualization=_build_section(VisualizationConfig, raw, "visualization"),
        cropping=_build_section(CroppingConfig, raw, "cropping"),
        config_path=config_path,
    )
