"""Image loading, bounding-box cropping, masking, and resolution equalization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def crop_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    padding: float = 0.0,
) -> Image.Image:
    """Crop a pixel-coordinate bounding box (x1, y1, x2, y2) with optional relative padding."""
    if bbox is None:
        return image

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box coordinates: {bbox}")

    if padding < 0:
        raise ValueError("padding must be non-negative.")

    width, height = image.size
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding

    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))

    if right <= left or bottom <= top:
        raise ValueError(f"Cropped box {bbox} lies outside image bounds {image.size}.")

    return image.crop((left, top, right, bottom))


def apply_foreground_mask(
    image: Image.Image,
    mask_path: str | Path | None,
    background_value: int = 0,
) -> Image.Image:
    """Zero out background pixels outside a binary foreground mask."""
    if mask_path is None or str(mask_path).strip() == "":
        return image

    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Foreground mask not found: {path}")

    with Image.open(path) as mask_handle:
        mask = mask_handle.convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    rgb = image.convert("RGB")
    img_arr = np.asarray(rgb, dtype=np.uint8)
    mask_arr = np.asarray(mask) > 0

    out = np.full_like(img_arr, fill_value=background_value, dtype=np.uint8)
    out[mask_arr] = img_arr[mask_arr]
    return Image.fromarray(out, mode="RGB")


def equalize_resolution(
    image: Image.Image,
    target_size: int,
    pad_value: int = 0,
) -> Image.Image:
    """Resize preserving aspect ratio and pad to a square target_size x target_size canvas."""
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    rgb = image.convert("RGB")
    width, height = rgb.size
    if width == target_size and height == target_size:
        return rgb

    scale = target_size / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = rgb.resize((new_w, new_h), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), (pad_value, pad_value, pad_value))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def _resolve_path(path_str: str | Path, raw_root: Path | None) -> Path:
    path = Path(path_str)
    if path.is_file() or raw_root is None:
        return path
    candidate = raw_root / path
    if candidate.is_file():
        return candidate
    return path


def _extract_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    keys = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    values: list[float] = []
    for key in keys:
        val = row.get(key)
        if val is None or pd.isna(val) or str(val).strip() == "":
            return None
        values.append(float(val))
    return (values[0], values[1], values[2], values[3])


def load_image_sample(
    row: Mapping[str, Any],
    *,
    raw_root: Path | None = None,
    use_foreground_mask: bool = False,
    equalize_size: int | None = None,
    crop_from_bbox: bool = True,
    bbox_padding: float = 0.0,
) -> Image.Image:
    """Load one manifest row as an RGB PIL image with optional crop, mask, and resize."""
    raw_image_path = row.get("image_path")
    if raw_image_path is None or pd.isna(raw_image_path) or str(raw_image_path).strip() == "":
        raise ValueError("Manifest row is missing image_path.")

    image_path = _resolve_path(str(raw_image_path), raw_root)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    if crop_from_bbox:
        bbox = _extract_bbox(row)
        if bbox is not None:
            image = crop_bbox(image, bbox, padding=bbox_padding)

    if use_foreground_mask:
        raw_mask = row.get("mask_path")
        if raw_mask is not None and not pd.isna(raw_mask) and str(raw_mask).strip() != "":
            mask_path = _resolve_path(str(raw_mask), raw_root)
            image = apply_foreground_mask(image, mask_path)

    if equalize_size is not None:
        image = equalize_resolution(image, equalize_size)

    return image
