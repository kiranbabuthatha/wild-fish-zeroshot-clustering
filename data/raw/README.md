# Raw Datasets Setup Guide

All raw fish images, videos, and archives live locally in `data/raw/<dataset>/` and are **ignored by `.gitignore`** so they are never pushed to GitHub.

## 1. Check Local Dataset Status

From the repository root, run:

```bash
python scripts/download_datasets.py --list
```

## 2. Automated Public Downloads

You can download public datasets hosted on Zenodo, Fish4Knowledge (`FRGT`), and Kaggle with:

```bash
# Download a single dataset or specific subset
python scripts/download_datasets.py --dataset frgt --dataset fishwio --dataset fjordfish

# Download all automated public datasets
python scripts/download_datasets.py --all-auto
```

Notes for Kaggle (`brackish`): install `kaggle` (`pip install kaggle`) and place your API token at `~/.kaggle/kaggle.json` before running `--dataset brackish`.

## 3. Dataset Inventory and Wild-Only Scope

| Folder | Role | Region | Source / Access | Wild-Only & Literature Notes |
|---|---|---|---|---|
| `frgt/` | RQ1, RQ2 | Taiwan reefs | Fish4Knowledge Ground Truth (`fishRecognition_GT.tar`) | `27,370` fish crops + `27,370` binary foreground masks across `23` classes; inspect sequence duplicates. |
| `ozfish/` | RQ1, RQ2 | Australia (BRUVS) | AIMS Repository (`10.25845/5e28f062c5097` / `github.com/open-AIMS/ozfish`) | `80,823` crops across `500+` taxa; keep left/right stereo pairs grouped in sampling. |
| `fishwio/` | RQ1, RQ2 | Mayotte, Indian Ocean | Zenodo (`10.5281/zenodo.18600403`) | `114,687` reef fish crops across `124` species + `8` background classes; cap samples per video. |
| `mediterranean/` | RQ1, RQ2 | Mallorca coast | Zenodo (`10.5281/zenodo.7534425`) | `1,590` coastal frames, `~18,500` YOLO boxes across `~20` species. |
| `fjordfish/` | RQ1, RQ2 | Norway & Sweden fjords | Zenodo (`10.5281/zenodo.17950781`) | `27` North Atlantic classes; **exclude IMR Flodevigen mesocosm cod frames** to keep strictly wild-only scope. |
| `brackish/` | RQ2 | Limfjorden, Denmark | Kaggle (`aalborguniversity/brackish-dataset`) | `15,084` temperate turbid frames (`6` coarse classes, no species IDs); useful optical bridge toward Baltic Sea. |
| `deepfish/` | RQ2 | Tropical Australia | Official repo (`github.com/alzayats/DeepFish`) | `~40k` whole habitat frames without species labels; used in full-frame vs. crop background tests. |
| `fishinturbidwater/` | Optional | Australia estuaries | Elsevier (`10.1016/j.ecoinf.2023.102303`) | `51` turbid underwater videos with clip-level labels for turbidity stress tests. |
| `hodor/` | RQ2 | Kiel Fjord, Baltic Sea | PANGAEA (`10.1594/PANGAEA.980000`) via `hodor-python` | German Baltic stereo observatory; use `hodor-python` to pull fish-active sequences instead of 400+ hours of raw video. |
| `schufi/` | Target | Baltic Sea | Partner imagery (University of Rostock / SCHUFI) | Unboxed Baltic monitoring frames; cropped via `scripts/crop_schufi_frames.py`. |

## 4. Intentionally Excluded Sets

- **`FishDet-M`:** Excluded from wild clustering because it merges wild marine datasets with aquarium tank photos and web-scraped `FishNet` images, and carries only a single `'fish'` detection class. Its detector benchmarks are referenced only for cropping `SCHUFI` frames.
- **`f4k_subset`:** Excluded as a smaller duplicate subset already covered by `frgt/`.
