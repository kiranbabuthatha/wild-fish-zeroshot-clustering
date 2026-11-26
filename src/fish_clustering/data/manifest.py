"""Dataset manifest builders normalizing wild fish datasets to a common schema."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from PIL import Image

LOGGER = logging.getLogger(__name__)

MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "source_type",
    "image_path",
    "mask_path",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "label_raw",
    "family",
    "genus",
    "species",
    "is_species_level",
    "video_id",
    "split",
]

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_FISHWIO_STAGE_SUFFIXES = (
    "_juvenile",
    "_adult",
    "_adu",
    "_female",
    "_male",
    "_initial",
    "_terminal",
    "_phase",
)
_NON_SPECIES_TOKENS = {
    "sp",
    "sp.",
    "sp-",
    "spp",
    "spp.",
    "unid",
    "unid-",
    "unknown",
    "na",
    "small",
    "taca",
    "poly",
    "prob",
    "fish",
    "small_fish",
}
_BRACKISH_FISH_TAGS = {"fish", "small_fish"}


def _to_manifest_path(path: Path | None, raw_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(raw_root).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_frgt_key(stem: str) -> str:
    clean = stem.strip()
    if clean.startswith("fish_"):
        clean = clean[len("fish_") :]
    elif clean.startswith("mask_"):
        clean = clean[len("mask_") :]
    if "_" in clean:
        left, right = clean.split("_", 1)
        left_norm = left.lstrip("0") or "0"
        return f"{left_norm}_{right}"
    return clean.lstrip("0") or "0"


def _parse_binomial(text: str) -> tuple[str, str, bool]:
    """Return (genus, full_species, is_species_level) when text is a Latin binomial."""
    cleaned = text.strip()
    if not cleaned:
        return "", "", False

    lowered = cleaned.lower()
    if any(tok in lowered.split() for tok in _NON_SPECIES_TOKENS):
        return "", "", False
    if any(marker in lowered for marker in ("sp-", "unid-", "prob ", "afish")):
        return "", "", False

    tokens = cleaned.replace("_", " ").split()
    if len(tokens) < 2:
        return "", "", False

    genus, epithet = tokens[0], tokens[1]
    if not genus[0].isupper():
        return "", "", False
    if epithet.lower() in _NON_SPECIES_TOKENS or not epithet.isalpha():
        return "", "", False

    return genus, f"{genus} {epithet.lower()}", True


def _parse_fishwio_folder(folder_name: str) -> tuple[str, str, str, bool]:
    base = folder_name
    for suffix in _FISHWIO_STAGE_SUFFIXES:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    genus, species, is_sp = _parse_binomial(base)
    return folder_name, genus, species, is_sp


def _make_row(
    *,
    image_id: str,
    dataset: str,
    source_type: str,
    image_path: str,
    mask_path: str = "",
    bbox: tuple[float, float, float, float] | None = None,
    label_raw: str = "",
    family: str = "",
    genus: str = "",
    species: str = "",
    is_species_level: bool = False,
    video_id: str = "",
    split: str = "",
) -> dict[str, object]:
    x1, y1, x2, y2 = bbox if bbox is not None else (None, None, None, None)
    return {
        "image_id": image_id,
        "dataset": dataset,
        "source_type": source_type,
        "image_path": image_path,
        "mask_path": mask_path,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "label_raw": label_raw,
        "family": family,
        "genus": genus,
        "species": species,
        "is_species_level": is_species_level,
        "video_id": video_id,
        "split": split,
    }


def _empty_manifest() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def _index_files(root: Path, extensions: set[str] | None = None) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if not root.is_dir():
        return indexed
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if extensions is not None and path.suffix.lower() not in extensions:
            continue
        indexed[path.name] = path
    return indexed


def build_frgt_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for FRGT cropped fish and foreground masks."""
    ds_root = raw_root / "frgt"
    csv_path = ds_root / "class_id.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"FRGT metadata file not found: {csv_path}")

    fish_dir = ds_root / "extracted" / "fish"
    if not fish_dir.is_dir():
        fish_dir = ds_root / "fish"
    if not fish_dir.is_dir():
        raise FileNotFoundError(f"FRGT fish image directory not found under {ds_root}")

    mask_dir = ds_root / "extracted" / "mask"
    if not mask_dir.is_dir():
        mask_dir = ds_root / "mask"

    fish_index: dict[str, Path] = {}
    for path in sorted(fish_dir.rglob("*.png")):
        fish_index[path.stem] = path
        fish_index[_normalize_frgt_key(path.stem)] = path

    mask_index: dict[str, Path] = {}
    if mask_dir.is_dir():
        for path in sorted(mask_dir.rglob("*.png")):
            mask_index[path.stem] = path
            mask_index[_normalize_frgt_key(path.stem)] = path

    meta = pd.read_csv(csv_path, dtype=str)
    meta.columns = [col.strip() for col in meta.columns]

    rows: list[dict[str, object]] = []
    for raw_name, raw_cluster in zip(meta["file_name"], meta["cluster_id"]):
        file_name = str(raw_name).strip()
        cluster_id = str(raw_cluster).strip()
        norm_key = _normalize_frgt_key(file_name)

        fish_path = fish_index.get(file_name) or fish_index.get(norm_key)
        if fish_path is None:
            continue

        mask_path = mask_index.get(f"mask_{file_name}") or mask_index.get(norm_key)
        rows.append(
            _make_row(
                image_id=f"frgt_{file_name}",
                dataset="frgt",
                source_type="wild",
                image_path=_to_manifest_path(fish_path, raw_root),
                mask_path=_to_manifest_path(mask_path, raw_root),
                label_raw=f"cluster_{cluster_id}",
                is_species_level=False,
            )
        )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _extract_ozfish_video_id(file_name: str) -> str:
    prefix = file_name.split(".", 1)[0]
    return re.sub(r"_[LR]$", "", prefix)


def build_ozfish_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for OzFish benthic BRUV fish crops."""
    ds_root = raw_root / "ozfish"
    csv_path = ds_root / "metadata" / "crop_metadata.csv"
    crops_dir = ds_root / "crops"
    if not csv_path.is_file():
        raise FileNotFoundError(f"OzFish metadata file not found: {csv_path}")
    if not crops_dir.is_dir():
        raise FileNotFoundError(f"OzFish crops directory not found: {crops_dir}")

    crop_files = {p.name: p for p in crops_dir.iterdir() if p.is_file()}
    meta = pd.read_csv(csv_path, dtype=str).fillna("")

    rows: list[dict[str, object]] = []
    for rec in meta.to_dict(orient="records"):
        file_name = rec["file_name"].strip()
        crop_path = crop_files.get(file_name)
        if crop_path is None:
            continue

        family = rec.get("family", "").strip()
        genus = rec.get("genus", "").strip()
        epithet = rec.get("species", "").strip()

        full_species = ""
        is_sp = False
        if genus and epithet and epithet.lower() not in _NON_SPECIES_TOKENS and epithet.isalpha():
            genus_clean = genus[0].upper() + genus[1:].lower()
            full_species = f"{genus_clean} {epithet.lower()}"
            genus = genus_clean
            is_sp = True
        else:
            genus_clean = genus[0].upper() + genus[1:].lower() if genus else ""
            genus = genus_clean

        label_raw = full_species or genus or family or "unknown"
        uid = rec.get("uid", "").strip() or Path(file_name).stem
        rows.append(
            _make_row(
                image_id=f"ozfish_{uid}",
                dataset="ozfish",
                source_type="wild",
                image_path=_to_manifest_path(crop_path, raw_root),
                label_raw=label_raw,
                family=family,
                genus=genus,
                species=full_species,
                is_species_level=is_sp,
                video_id=_extract_ozfish_video_id(file_name),
            )
        )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _extract_fishwio_video_id(stem: str) -> str:
    # Crop filenames follow <video_or_site>_<hash>_<frame>_<w>x<h>+<x>+<y>.
    match = re.match(r"^(.+?)_[0-9a-f]{8,}_\d+_\d+x\d+", stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem


def build_fishwio_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for FishWIO Western Indian Ocean fish crops."""
    ds_root = raw_root / "fishwio"
    data_dir = ds_root / "extracted" / "data_paper_dataset"
    if not data_dir.is_dir():
        data_dir = ds_root / "data_paper_dataset"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"FishWIO dataset directory not found under {ds_root}")

    rows: list[dict[str, object]] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        label_raw, genus, species, is_sp = _parse_fishwio_folder(class_dir.name)
        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            rows.append(
                _make_row(
                    image_id=f"fishwio_{class_dir.name}_{img_path.stem}",
                    dataset="fishwio",
                    source_type="wild",
                    image_path=_to_manifest_path(img_path, raw_root),
                    label_raw=label_raw,
                    genus=genus,
                    species=species,
                    is_species_level=is_sp,
                    video_id=_extract_fishwio_video_id(img_path.stem),
                )
            )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_mediterranean_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows from DeepEcoMar Mediterranean YOLO detection annotations."""
    ds_root = raw_root / "mediterranean"
    yolo_root = ds_root / "extracted" / "DEEPECOMAR_DATASET_YOLO"
    if not yolo_root.is_dir():
        yolo_root = ds_root / "DEEPECOMAR_DATASET_YOLO"
    data_yaml = yolo_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Mediterranean YOLO data.yaml not found under {ds_root}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)
    names = meta.get("names", [])
    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        class_names = {idx: str(name) for idx, name in enumerate(names)}

    rows: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        img_dir = yolo_root / split / "images"
        lbl_dir = yolo_root / split / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue

        img_by_stem = {
            p.stem: p
            for p in sorted(img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        }
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            img_path = img_by_stem.get(lbl_path.stem)
            if img_path is None:
                continue
            lines = [line.strip() for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                continue

            with Image.open(img_path) as im:
                width, height = im.size

            video_id = re.split(r"_(?:frame|jpg)", img_path.stem, maxsplit=1)[0]
            for box_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                x1 = max(0.0, (cx - bw / 2.0) * width)
                y1 = max(0.0, (cy - bh / 2.0) * height)
                x2 = min(float(width), (cx + bw / 2.0) * width)
                y2 = min(float(height), (cy + bh / 2.0) * height)
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = class_names.get(cls_id, str(cls_id))
                genus, species, is_sp = _parse_binomial(class_name)
                rows.append(
                    _make_row(
                        image_id=f"med_{split}_{img_path.stem}_{box_idx}",
                        dataset="mediterranean",
                        source_type="wild",
                        image_path=_to_manifest_path(img_path, raw_root),
                        bbox=(x1, y1, x2, y2),
                        label_raw=class_name,
                        genus=genus,
                        species=species,
                        is_species_level=is_sp,
                        video_id=video_id,
                        split=split,
                    )
                )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_brackish_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows from AAU Brackish bounding-box annotations (fish tags only)."""
    ds_root = raw_root / "brackish"
    ann_dir = ds_root / "extracted" / "annotations" / "annotations_AAU"
    frames_dir = ds_root / "extracted" / "frames"
    if not ann_dir.is_dir():
        ann_dir = ds_root / "annotations" / "annotations_AAU"
    if not frames_dir.is_dir():
        frames_dir = ds_root / "frames"
    if not ann_dir.is_dir() or not frames_dir.is_dir():
        raise FileNotFoundError(f"Brackish annotations or frames directory missing under {ds_root}")

    frame_files = _index_files(frames_dir, _IMAGE_EXTENSIONS)
    rows: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        csv_path = ann_dir / f"{split}.csv"
        if not csv_path.is_file():
            continue

        ann_df = pd.read_csv(csv_path, sep=None, engine="python", dtype=str)
        ann_df.columns = [col.strip() for col in ann_df.columns]
        for row_idx, rec in enumerate(ann_df.to_dict(orient="records")):
            tag = str(rec.get("Annotation tag", "")).strip()
            if tag.lower() not in _BRACKISH_FISH_TAGS:
                continue

            fname = str(rec.get("Filename", "")).strip()
            frame_path = frame_files.get(fname)
            if frame_path is None:
                continue

            x1 = float(rec["Upper left corner X"])
            y1 = float(rec["Upper left corner Y"])
            x2 = float(rec["Lower right corner X"])
            y2 = float(rec["Lower right corner Y"])
            if x2 <= x1 or y2 <= y1:
                continue

            obj_id = str(rec.get("Object ID", row_idx)).strip()
            video_id = Path(fname).stem.rsplit("-", 1)[0]
            rows.append(
                _make_row(
                    image_id=f"brackish_{split}_{Path(fname).stem}_{obj_id}_{row_idx}",
                    dataset="brackish",
                    source_type="wild",
                    image_path=_to_manifest_path(frame_path, raw_root),
                    bbox=(x1, y1, x2, y2),
                    label_raw=tag,
                    is_species_level=False,
                    video_id=video_id,
                    split=split,
                )
            )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_deepfish_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows from DeepFish Classification positive fish frames."""
    ds_root = raw_root / "deepfish"
    cls_dir = ds_root / "extracted" / "DeepFish" / "Classification"
    if not cls_dir.is_dir():
        cls_dir = ds_root / "Classification"
    if not cls_dir.is_dir():
        raise FileNotFoundError(f"DeepFish Classification directory not found under {ds_root}")

    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        csv_path = cls_dir / f"{split}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for rec in df.to_dict(orient="records"):
            if rec.get("labels", "").strip() != "1":
                continue
            rel_id = rec["ID"].strip()
            img_path: Path | None = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = cls_dir / f"{rel_id}{ext}"
                if candidate.is_file():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            habitat = rec.get("habitats", "").strip()
            rows.append(
                _make_row(
                    image_id=f"deepfish_{rel_id.replace('/', '_')}",
                    dataset="deepfish",
                    source_type="wild",
                    image_path=_to_manifest_path(img_path, raw_root),
                    label_raw=f"habitat_{habitat}" if habitat else "fish",
                    is_species_level=False,
                    video_id=habitat,
                    split=split,
                )
            )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _build_class_folder_dataset(
    raw_root: Path,
    dataset_name: str,
    folder_candidates: list[Path],
    source_type: str,
) -> pd.DataFrame:
    data_dir = next((p for p in folder_candidates if p.is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} directory not found; checked {[str(p) for p in folder_candidates]}"
        )

    rows: list[dict[str, object]] = []
    for img_path in sorted(data_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        label_folder = img_path.parent.name
        genus, species, is_sp = _parse_binomial(label_folder)
        rows.append(
            _make_row(
                image_id=f"{dataset_name}_{label_folder}_{img_path.stem}",
                dataset=dataset_name,
                source_type=source_type,
                image_path=_to_manifest_path(img_path, raw_root),
                label_raw=label_folder,
                genus=genus,
                species=species,
                is_species_level=is_sp,
            )
        )

    if not rows:
        return _empty_manifest()
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_fjordfish_wild_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for Norwegian fjord wild underwater images."""
    return _build_class_folder_dataset(
        raw_root,
        "fjordfish_wild",
        [
            raw_root / "fjordfish_wild" / "extracted",
            raw_root / "fjordfish_wild",
            raw_root / "fjordfish" / "extracted",
            raw_root / "fjordfish",
        ],
        source_type="wild",
    )


def build_hodor_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for Baltic HODOR cod and flatfish underwater images."""
    return _build_class_folder_dataset(
        raw_root,
        "hodor",
        [raw_root / "hodor" / "extracted", raw_root / "hodor"],
        source_type="wild",
    )


def build_schufi_manifest(raw_root: Path) -> pd.DataFrame:
    """Build manifest rows for SCHUFI studio/trawl reference images."""
    return _build_class_folder_dataset(
        raw_root,
        "schufi",
        [raw_root / "schufi" / "extracted", raw_root / "schufi"],
        source_type="controlled",
    )


DATASET_BUILDERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "frgt": build_frgt_manifest,
    "ozfish": build_ozfish_manifest,
    "fishwio": build_fishwio_manifest,
    "mediterranean": build_mediterranean_manifest,
    "brackish": build_brackish_manifest,
    "deepfish": build_deepfish_manifest,
    "fjordfish_wild": build_fjordfish_wild_manifest,
    "hodor": build_hodor_manifest,
    "schufi": build_schufi_manifest,
}


def build_manifest(
    raw_root: str | Path,
    datasets: list[str],
    *,
    skip_missing: bool = False,
) -> pd.DataFrame:
    """Build and concatenate normalized manifest tables across requested datasets."""
    root = Path(raw_root)
    frames: list[pd.DataFrame] = []

    for name in datasets:
        builder = DATASET_BUILDERS.get(name)
        if builder is None:
            known = sorted(DATASET_BUILDERS.keys())
            raise ValueError(f"Unknown dataset {name!r}; supported datasets: {known}")
        try:
            df = builder(root)
        except FileNotFoundError as exc:
            if skip_missing:
                LOGGER.warning("Skipping missing dataset %s: %s", name, exc)
                continue
            raise
        if not df.empty:
            frames.append(df)

    if not frames:
        return _empty_manifest()
    return pd.concat(frames, ignore_index=True)
