# Zero-Shot Clustering of Wild Fish Images Using Vision Transformers

**Master's Thesis Codebase (OVGU Magdeburg & SCHUFI, University of Rostock)**  
**Author:** Kiran Babu Thatha (M.Sc. Digital Engineering, OVGU Magdeburg)  
**Supervisors:** Prof. Dr.-Ing. habil. Holger Theisel (OVGU Magdeburg), Dr. Vivian Helen Fischbach & Sai Dheeraj Reddy Tippani (SCHUFI Project, University of Rostock)

## Overview

This repository contains the base pipeline for evaluating frozen Vision Transformers on wild marine and brackish fish imagery without task-specific training or fine-tuning. Building on Markoff, Bengtson & Ørsted (2026) as the reference study, the codebase is structured around two research questions:

- **RQ1 (Transfer to Wild Fish):** How well do frozen Vision Transformer representations (`DINOv3`, `BioCLIP 2`, `AquaticCLIP`), dimensionality reduction (`PCA`, `openTSNE`, `UMAP`), and clustering methods (`HDBSCAN`, `Ward`, `k-means`, `GMM`) transfer from terrestrial wildlife benchmarks to open wild marine and brackish fish datasets?
- **RQ2 (SCHUFI & Cross-Dataset Relation):** When open fish datasets and `SCHUFI` Baltic Sea images share the same embedding space, do species group together across datasets, or do images separate by camera, resolution, background, turbidity, or novel local Baltic species?

Key methodological controls built into the pipeline:
- Video/track near-duplicate sampling caps and wild-only dataset filtering (e.g., removing mesocosm cod from `FjordFish` and excluding captive/web subsets from `FishDet-M`).
- Optional foreground mask check (`FRGT` binary masks) and equalized-resolution control runs.
- Dual-space evaluation: clustering in 2-D projections and L2-normalized high-D embeddings checked with `Trustworthiness` and `Continuity`, while `RQ2` nearest-neighbor mixing and OOD scoring run directly in the high-dimensional embedding space.

## Repository Structure

```text
Base_code_Thesis/
├── configs/                  # Experiment YAML configs (default + smoke test)
├── data/
│   ├── raw/                  # Local raw datasets (not tracked by Git)
│   ├── manifests/            # Standardized dataset CSV manifests (local)
│   └── processed/            # Cropped SCHUFI/HODOR frames & masked subsets (local)
├── outputs/
│   ├── embeddings/           # Cached L2-normalized backbone embeddings
│   ├── metrics/              # RQ1, RQ2, and projection quality JSON/CSV tables
│   └── visual_atlas/         # Interactive HTML embedding explorer plots
├── scripts/                  # CLI entry points (manifests, cropping, main pipeline)
├── src/fish_clustering/      # Core Python package
│   ├── config.py             # YAML config loader
│   ├── data/                 # Manifest builders, samplers, and mask/resize transforms
│   ├── models/               # Frozen ViT / VLM feature extractors and pooling
│   ├── clustering/           # PCA, openTSNE (out-of-sample), UMAP, HDBSCAN/Ward/GMM
│   ├── evaluation/           # RQ1 clustering, Trustworthiness/Continuity, DBCV, RQ2 OOD
│   └── visualization/        # Interactive 2-D visual atlas with distortion/OOD flags
└── tests/                    # Unit tests
```

## Setup

Python 3.10 or 3.11 is recommended:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quickstart

1. Build unified dataset manifests from local `data/raw/` folders:

```bash
python scripts/build_manifests.py --config configs/smoke_test.yaml
```

2. Run the clustering, evaluation, interactive explorer, and static figure pipeline (add `--extractor-backend dummy` for fast offline smoke checks without GPU checkpoints):

```bash
python scripts/run_pipeline.py --config configs/smoke_test.yaml
```

3. View the results locally:
   - **Interactive Web Explorer (Markoff 2026 style with fish thumbnails, high-D neighbors, centroids/hulls, distortion, and OOD overlay):**
     ```bash
     open outputs/visual_atlas/frgt_smoke_test/explorer/index.html
     ```
     Or serve via local HTTP server:
     ```bash
     python -m http.server 8000 --directory outputs/visual_atlas/frgt_smoke_test/explorer
     ```
   - **Static 300-DPI Thesis Figures:** inspect `outputs/visual_atlas/frgt_smoke_test/figures/*.png`
   - **Metric Tables:** inspect `outputs/metrics/frgt_smoke_test/summary.json`, `clustering_metrics.csv`, and `projection_metrics.csv`

4. Run the unit test suite:

```bash
pytest -v
```

## Local Data Layout

Raw fish datasets and cached embeddings are kept local only and ignored by Git. Place open-source datasets under `data/raw/<dataset_name>/` before building manifests:

- `data/raw/frgt/` (`FRGT` crops and binary masks)
- `data/raw/ozfish/` (`OzFish` crops)
- `data/raw/fishwio/` (`FishWIO` crops and video metadata)
- `data/raw/mediterranean/` (Coastal Mediterranean boxes)
- `data/raw/fjordfish/` (`FjordFish`, filtered to wild RUV frames only)
- `data/raw/brackish/` (`Brackish` Limfjorden dataset)
- `data/raw/deepfish/` (`DeepFish` full habitat frames)
- `data/raw/hodor/` (`HODOR` Baltic Sea Kiel Fjord frames/crops)
- `data/raw/schufi/` (`SCHUFI` Baltic Sea target frames)

### Downloading Datasets

Check which datasets are already present locally or fetch public datasets into `data/raw/` using [`scripts/download_datasets.py`](scripts/download_datasets.py):

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --dataset frgt --dataset fishwio --dataset fjordfish
```

See [`data/raw/README.md`](data/raw/README.md) for source links, manual repositories, and wild-only filtering notes.

## Development Plan

1. **Initial Structure (done):** Clean folder layout, dependency setup, and YAML configs.
2. **Data & Configs (done):** Typed YAML config loader, unified manifest schema, video-cap sampler, background mask toggle, and resolution control.
3. **Feature Extraction & Reduction (done):** Frozen `DINOv3` (`CLS` and `patch_mean` pooling) + `BioCLIP 2` wrapper, embedding cache, and `openTSNE` out-of-sample projection.
4. **Metrics, Visual Atlas & Pipeline (done):** `V-measure`/`ARI`/`AMI`/`DBCV`/`silhouette`, `Trustworthiness`/`Continuity`, high-D `kNN` mixing & OOD scores, interactive HTML atlas, runnable `scripts/run_pipeline.py`, and `pytest` unit tests.
5. **Frame Cropping (pending detector confirmation):** Wire `scripts/crop_schufi_frames.py` once `Grounding DINO` vs. `FishDet-M` is chosen on Baltic Sea frames.
