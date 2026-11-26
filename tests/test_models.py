"""Unit tests for feature extractor registry, batch extraction, and disk cache."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fish_clustering.models import (
    BACKBONE_SPECS,
    BaseFeatureExtractor,
    build_cache_key,
    create_extractor,
    extract_embeddings,
    extract_or_load_embeddings,
    get_backbone_spec,
    l2_normalize_embeddings,
    list_supported_backbones,
)


class DummyExtractor(BaseFeatureExtractor):
    """Deterministic lightweight extractor returning padded image RGB means."""

    def __init__(self, *, pooling: str = "cls", dim: int = 16, image_size: int = 32) -> None:
        super().__init__(
            "dummy_vit",
            pooling=pooling,
            device="cpu",
            image_size=image_size,
        )
        self.dim = dim
        self.calls = 0

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        self.calls += 1
        rows = []
        for img in images:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            mean_rgb = arr.mean(axis=(0, 1)) / 255.0
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[:3] = mean_rgb
            vec[3] = 1.0
            rows.append(vec)
        return np.stack(rows, axis=0)


def _write_sample(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def test_backbone_registry_and_validation() -> None:
    supported = list_supported_backbones()
    assert "dinov3_vitb16" in supported
    assert "dinov3_vitl16" in supported
    assert "bioclip2_vitl14" in supported
    assert len(supported) == len(BACKBONE_SPECS)

    spec = get_backbone_spec("bioclip2_vitl14")
    assert spec.family == "open_clip"
    assert spec.model_id == "hf-hub:imageomics/bioclip-2"

    dinov3_spec = get_backbone_spec("dinov3_vitb16")
    assert dinov3_spec.hf_model_id == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    with pytest.raises(ValueError, match="Unsupported backbone"):
        get_backbone_spec("nonexistent_model")

    with pytest.raises(ValueError, match="Unsupported pooling"):
        create_extractor("dinov3_vitb16", pooling="invalid")


def test_l2_normalize_embeddings() -> None:
    mat = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    normed = l2_normalize_embeddings(mat)
    assert normed.dtype == np.float32
    assert np.allclose(np.linalg.norm(normed[0]), 1.0, atol=1e-6)
    assert np.allclose(normed[0], [0.6, 0.8, 0.0], atol=1e-6)
    assert np.allclose(normed[1], [0.0, 0.0, 0.0], atol=1e-6)


def test_extract_embeddings_and_cache_lifecycle(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    mask_path = raw_root / "frgt" / "m0.png"
    _write_sample(raw_root / "frgt" / "f0.png", (200, 40, 40))
    _write_sample(raw_root / "frgt" / "f1.png", (40, 200, 40))
    _write_sample(raw_root / "frgt" / "f2.png", (40, 40, 200))

    mask_arr = np.zeros((24, 24), dtype=np.uint8)
    mask_arr[4:20, 4:20] = 255
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_arr, mode="L").save(mask_path)

    manifest = pd.DataFrame(
        [
            {"image_id": "id_0", "image_path": "frgt/f0.png", "mask_path": "frgt/m0.png"},
            {"image_id": "id_1", "image_path": "frgt/f1.png", "mask_path": ""},
            {"image_id": "id_2", "image_path": "frgt/f2.png", "mask_path": ""},
        ]
    )

    extractor = DummyExtractor(dim=16, image_size=32)
    embs, ids = extract_embeddings(
        manifest,
        extractor,
        raw_root=raw_root,
        batch_size=2,
        l2_normalize=True,
        use_foreground_mask=True,
        equalize_size=32,
    )
    assert ids == ["id_0", "id_1", "id_2"]
    assert embs.shape == (3, 16)
    assert extractor.calls == 2
    assert np.allclose(np.linalg.norm(embs, axis=1), 1.0, atol=1e-5)

    cache_dir = tmp_path / "cache"
    extractor.calls = 0
    first = extract_or_load_embeddings(
        manifest,
        extractor,
        cache_dir=cache_dir,
        raw_root=raw_root,
        batch_size=2,
        use_foreground_mask=False,
    )
    assert extractor.calls == 2
    assert first.cache_path is not None and first.cache_path.is_file()

    # Second call with same manifest and flags must hit disk cache.
    second = extract_or_load_embeddings(
        manifest,
        extractor,
        cache_dir=cache_dir,
        raw_root=raw_root,
        batch_size=2,
        use_foreground_mask=False,
    )
    assert extractor.calls == 2
    assert np.allclose(first.embeddings, second.embeddings)
    assert first.image_ids == second.image_ids

    # force_recompute triggers a fresh pass through the extractor.
    third = extract_or_load_embeddings(
        manifest,
        extractor,
        cache_dir=cache_dir,
        raw_root=raw_root,
        batch_size=2,
        use_foreground_mask=False,
        force_recompute=True,
    )
    assert extractor.calls == 4
    assert third.embeddings.shape == (3, 16)

    key_nomask = build_cache_key(
        backbone="dinov3_vitb16",
        pooling="cls",
        image_size=224,
        l2_normalize=True,
        use_foreground_mask=False,
        equalize_resolution=None,
        image_ids=ids,
    )
    key_mask = build_cache_key(
        backbone="dinov3_vitb16",
        pooling="cls",
        image_size=224,
        l2_normalize=True,
        use_foreground_mask=True,
        equalize_resolution=None,
        image_ids=ids,
    )
    assert key_nomask != key_mask
