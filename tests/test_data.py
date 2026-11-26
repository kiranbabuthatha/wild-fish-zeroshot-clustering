"""Unit tests for dataset manifest builders, sampling filters, and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fish_clustering.data import (
    MANIFEST_COLUMNS,
    apply_foreground_mask,
    apply_sampling_filters,
    build_manifest,
    cap_crops_per_video,
    cap_images_per_species,
    crop_bbox,
    equalize_resolution,
    filter_shared_species,
    filter_species_level,
    load_image_sample,
)


def _write_rgb_image(path: Path, size: tuple[int, int] = (40, 20), color: tuple[int, int, int] = (100, 150, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_mask_image(path: Path, size: tuple[int, int] = (40, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[5:15, 10:30] = 255
    Image.fromarray(arr, mode="L").save(path)


def test_build_frgt_and_ozfish_and_fishwio_manifests(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"

    # Synthetic FRGT fixture matching zero-padded disk names vs unpadded CSV names.
    frgt_root = raw_root / "frgt"
    _write_rgb_image(frgt_root / "extracted" / "fish" / "fish_01" / "fish_000420000101_00001.png")
    _write_mask_image(frgt_root / "extracted" / "mask" / "mask_01" / "mask_000420000101_00001.png")
    (frgt_root / "class_id.csv").write_text("file_name,cluster_id\n00420000101_00001,3\n", encoding="utf-8")

    # Synthetic OzFish fixture with binomial species and stereo L/R naming.
    oz_root = raw_root / "ozfish"
    (oz_root / "metadata").mkdir(parents=True, exist_ok=True)
    _write_rgb_image(oz_root / "crops" / "A000001_L.avi.100.png")
    _write_rgb_image(oz_root / "crops" / "A000001_R.avi.100.png")
    (oz_root / "metadata" / "crop_metadata.csv").write_text(
        "uid,file_name,family,genus,species\n"
        "1,A000001_L.avi.100.png,Carangidae,Caranx,melampygus\n"
        "2,A000001_R.avi.100.png,Carangidae,Caranx,sp\n",
        encoding="utf-8",
    )

    # Synthetic FishWIO fixture with fish folder and underscored/hidden files.
    fw_root = raw_root / "fishwio" / "extracted" / "data_paper_dataset"
    _write_rgb_image(fw_root / "Caranx_melampygus" / "BA_19_J4_aabbccddeeff_10_32x24+0+0.jpeg")
    (fw_root / "Caranx_melampygus" / "._appledouble.jpeg").write_bytes(b"AppleDouble")
    _write_rgb_image(fw_root / "_sand" / "ignore_bg.jpeg")

    df = build_manifest(raw_root, ["frgt", "ozfish", "fishwio"])
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 4

    frgt_row = df[df["dataset"] == "frgt"].iloc[0]
    assert frgt_row["mask_path"].endswith("mask_000420000101_00001.png")
    assert frgt_row["label_raw"] == "cluster_3"
    assert bool(frgt_row["is_species_level"]) is False

    oz_rows = df[df["dataset"] == "ozfish"]
    assert oz_rows["video_id"].tolist() == ["A000001", "A000001"]
    assert oz_rows.iloc[0]["species"] == "Caranx melampygus"
    assert bool(oz_rows.iloc[0]["is_species_level"]) is True
    assert bool(oz_rows.iloc[1]["is_species_level"]) is False

    fw_rows = df[df["dataset"] == "fishwio"]
    assert len(fw_rows) == 1
    assert fw_rows.iloc[0]["species"] == "Caranx melampygus"

    shared = filter_shared_species(df, min_datasets=2)
    assert set(shared["dataset"]) == {"ozfish", "fishwio"}
    assert set(shared["species"]) == {"Caranx melampygus"}


def test_build_mediterranean_brackish_deepfish_and_skip_missing(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"

    # Synthetic Mediterranean YOLO fixture.
    med_root = raw_root / "mediterranean" / "extracted" / "DEEPECOMAR_DATASET_YOLO"
    (med_root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (med_root / "data.yaml").write_text(
        "nc: 2\nnames: [Chromis chromis, Diplodus sp-]\n",
        encoding="utf-8",
    )
    _write_rgb_image(med_root / "train" / "images" / "CAM1_frame_10.jpg", size=(100, 80))
    (med_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (med_root / "train" / "labels" / "CAM1_frame_10.txt").write_text(
        "0 0.5 0.5 0.4 0.5\n1 0.2 0.2 0.2 0.2\n",
        encoding="utf-8",
    )

    # Synthetic Brackish AAU semicolon-delimited fixture (filters non-fish tags).
    br_root = raw_root / "brackish" / "extracted"
    _write_rgb_image(br_root / "frames" / "clip1-0001.png", size=(100, 80))
    (br_root / "annotations" / "annotations_AAU").mkdir(parents=True, exist_ok=True)
    (br_root / "annotations" / "annotations_AAU" / "train.csv").write_text(
        "Filename;Object ID;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y\n"
        "clip1-0001.png;1;fish;10;20;60;50\n"
        "clip1-0001.png;2;crab;5;5;20;20\n",
        encoding="utf-8",
    )

    # Synthetic DeepFish fixture (keeps only labels == 1).
    dp_root = raw_root / "deepfish" / "extracted" / "DeepFish" / "Classification"
    _write_rgb_image(dp_root / "99" / "valid" / "frame1.jpg", size=(64, 64))
    _write_rgb_image(dp_root / "99" / "empty" / "frame2.jpg", size=(64, 64))
    (dp_root / "train.csv").write_text(
        "ID,habitats,labels,classes,frames\n"
        "99/valid/frame1,99,1,valid,frame1\n"
        "99/empty/frame2,99,0,empty,frame2\n",
        encoding="utf-8",
    )

    df = build_manifest(
        raw_root,
        ["mediterranean", "brackish", "deepfish", "hodor"],
        skip_missing=True,
    )
    assert len(df) == 4
    med = df[df["dataset"] == "mediterranean"]
    assert med.iloc[0]["species"] == "Chromis chromis"
    assert med.iloc[0]["bbox_x1"] == pytest.approx(30.0)
    assert med.iloc[0]["bbox_x2"] == pytest.approx(70.0)
    assert bool(med.iloc[1]["is_species_level"]) is False

    br = df[df["dataset"] == "brackish"]
    assert len(br) == 1
    assert br.iloc[0]["label_raw"] == "fish"

    dp = df[df["dataset"] == "deepfish"]
    assert len(dp) == 1
    assert dp.iloc[0]["label_raw"] == "habitat_99"


def test_sampling_caps_and_filters() -> None:
    rows = []
    for i in range(10):
        rows.append(
            {
                "image_id": f"img_{i}",
                "dataset": "ozfish" if i < 8 else "fishwio",
                "source_type": "wild",
                "image_path": f"p_{i}.png",
                "mask_path": "",
                "bbox_x1": None,
                "bbox_y1": None,
                "bbox_x2": None,
                "bbox_y2": None,
                "label_raw": "Chromis chromis",
                "family": "Pomacentridae",
                "genus": "Chromis",
                "species": "Chromis chromis" if i != 9 else "",
                "is_species_level": i != 9,
                "video_id": "vid_A" if i < 6 else "vid_B",
                "split": "",
            }
        )
    df = pd.DataFrame(rows)

    vid_capped = cap_crops_per_video(df, max_per_video=2, seed=42)
    # ozfish vid_A capped to 2, ozfish vid_B has 2, fishwio vid_B has 2 -> 6 total.
    assert len(vid_capped) == 6
    assert (vid_capped.groupby(["dataset", "video_id"]).size() <= 2).all()

    sp_capped = cap_images_per_species(df, max_per_species=3, seed=42)
    assert (sp_capped[sp_capped["species"] == "Chromis chromis"].shape[0]) == 3

    only_sp = filter_species_level(df)
    assert len(only_sp) == 9

    combined = apply_sampling_filters(
        df,
        species_level_only=True,
        max_crops_per_video=2,
        max_images_per_species=3,
        seed=42,
    )
    assert len(combined) <= 3


def test_preprocessing_bbox_mask_and_equalize(tmp_path: Path) -> None:
    img_path = tmp_path / "raw" / "frgt" / "fish.png"
    mask_path = tmp_path / "raw" / "frgt" / "mask.png"
    _write_rgb_image(img_path, size=(60, 40), color=(200, 100, 50))
    _write_mask_image(mask_path, size=(30, 20))

    with Image.open(img_path) as img:
        cropped = crop_bbox(img, (10, 10, 40, 30))
        assert cropped.size == (30, 20)

        masked = apply_foreground_mask(cropped, mask_path)
        arr = np.asarray(masked)
        # Pixel (0, 0) is outside mask and must be zeroed; center (15, 10) stays foreground.
        assert tuple(arr[0, 0]) == (0, 0, 0)
        assert tuple(arr[10, 15]) == (200, 100, 50)

        sq = equalize_resolution(masked, target_size=64)
        assert sq.size == (64, 64)

    sample = load_image_sample(
        {
            "image_path": "frgt/fish.png",
            "mask_path": "frgt/mask.png",
            "bbox_x1": 10.0,
            "bbox_y1": 10.0,
            "bbox_x2": 40.0,
            "bbox_y2": 30.0,
        },
        raw_root=tmp_path / "raw",
        use_foreground_mask=True,
        equalize_size=64,
    )
    assert sample.size == (64, 64)
