# Project Progress Log and Iteration Tracker

Living log for the Master's thesis codebase (`wild-fish-zeroshot-clustering`, OVGU Magdeburg & SCHUFI, University of Rostock). Update this file after each experiment run, iteration milestone, or supervisor meeting.

---

## 1. Current Status Snapshot

- **Repository:** `kiranbabuthatha/wild-fish-zeroshot-clustering` (`main`)
- **Hardware & Runtime:** Apple M4 Pro (`arm64`), Python `3.11.18` (`.venv`), PyTorch `2.15.0`, `timm 1.0.31`, `transformers 5.24.1`
- **Unit Test Suite:** `20` passing tests (`pytest -v` covering config, data/manifests, sampling, models/cache, DR/clustering, evaluation metrics, visualization, and end-to-end pipeline)
- **Datasets Ready Locally (`data/raw/`):** `frgt`, `ozfish`, `fishwio`, `mediterranean`, `brackish`, `deepfish`, `fishinturbidwater`
- **Datasets Pending Acquisition:** `fjordfish_wild` (download + exclude IMR Flodevigen mesocosm cod), `hodor` (fetch via `hodor-python`), `schufi` (partner Baltic frames)

---

## 2. Completed Milestones

| Commit | Summary | Key Outputs |
| :---: | :--- | :--- |
| `bdee17a` | Initial project structure | Package skeleton, `.gitignore`, `pyproject.toml`, `configs/default.yaml`, `configs/smoke_test.yaml` |
| `8eab32b` | Dataset downloader & setup | `scripts/download_datasets.py`, `data/raw/README.md`, 7 local datasets mirrored |
| `666e141` | Configs, manifests & sampling | `config.py`, `data/manifest.py`, `data/sampling.py`, `data/preprocessing.py`, `scripts/build_manifests.py` |
| `1f8fc58` | Backbones, cache, DR & clustering | `DINOv3`/`BioCLIP 2`/`DINOv2` wrappers, `.npz` cache, `PCA`/`openTSNE`/`UMAP`, `HDBSCAN`/`Ward`/`k-means`/`GMM` |
| `3c55f88` | Metrics & end-to-end runner | `V-Measure`, `ARI`, `AMI`, `Silhouette`, `DBCV`, `Trustworthiness`, `Continuity`, RQ2 mixing/OOD, `scripts/run_pipeline.py` |
| `36d5697` | Interactive Web Explorer | WebGL `explorer/index.html` with `.webp` thumbnails, FRGT masks, Top-5 high-D neighbors, centroids/hulls, cluster modals |
| `cdcd944` | Static thesis figures & docs | `visualization/figures.py` (300-DPI PNG suite: thumbnail maps, dendrogram, k-sweep, projection quality, RQ2 plots) |
| `1d18268` | Dark mode & HF DINOv3 fallback | Explorer Dark/Light toggle (`localStorage`), gated HuggingFace `transformers` DINOv3 extractor, `configs/fishwio_dinov3_demo.yaml`, dotfile fix |
| `3705067` | Metrics & data documentation | `docs/metrics_and_methods.md` (with visual examples) and `docs/data_sources.md` |
| `9e7d897` | Sampling cap & balanced DINOv3 runs | Per-(video, species) capping, `min_images_per_species` and `top_n_species` filters, real DINOv3 ViT-B/16 benchmarks on `FishWIO` and `FRGT` |

---

## 3. Verified Experiment Results Log

| Date | Experiment (`configs/`) | Backbone / Pooling | Sample Scope | Best 2-D HDBSCAN (`V` / `ARI` / `k`) | Best Fixed-`k` 2-D (`Method` / `V` / `ARI`) | High-D Ward (`V` / `ARI`) | Proj. Quality (`t-SNE` T/C) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| 2026-11-26 | `fishwio_dinov3_demo` | `dinov3_vitb16` (`cls`, `768-D`) | `FishWIO` (`900` crops, `30` species, `20-30`/sp) | UMAP `0.837` / `0.646` (`k=31`) | UMAP Ward `0.870` / `0.747` | `0.809` / `0.600` | `0.967` / `0.950` |
| 2026-11-26 | `smoke_test` | `dinov3_vitb16` (`cls`, `768-D`) | `FRGT` (`456` crops, `23` classes, `<=20`/cls) | UMAP `0.823` / `0.626` (`k=27`) | UMAP GMM `0.839` / `0.671` | `0.795` / `0.573` | `0.976` / `0.969` |

*Add new rows above whenever running a new backbone (`dinov3_vitl16`, `dinov3_vith16plus`, `bioclip2_vitl14`), pooling (`patch_mean`), mask ablation, or multi-dataset RQ2 sweep.*

---

## 4. Iteration Backlog (Things to Implement Next)

### A. Backbone & Pooling Ablations (RQ1)
- [ ] Install `open-clip-torch` and `pyarrow` into `.venv` and verify `bioclip2_vitl14` (`hf-hub:imageomics/bioclip-2`) on `fishwio_dinov3_demo` and `frgt`.
- [ ] Benchmark `dinov3_vitl16` (`1024-D`) and `dinov3_vith16plus` (`1280-D`) on `FishWIO`, `FRGT`, and `OzFish`.
- [ ] Run `cls` vs. `patch_mean` pooling comparison across DINOv3 variants (test whether foreground patch averaging reduces reef background dominance).
- [ ] Evaluate `AquaticCLIP` alongside `BioCLIP 2` once checkpoint availability is confirmed.
- [ ] Run multi-seed evaluations (`seeds: [42, 43, 44]`) to report mean +/- std stability for DR and clustering.

### B. Methodological Controls (RQ1 & RQ2)
- [ ] Run `FRGT` foreground-mask ablation (`use_foreground_mask: false` vs. `true`) on identical sampled crops to quantify reef background leakage.
- [ ] Run equalized-resolution check (`equalize_resolution: 64` or `112`) on merged `OzFish` + `FishWIO` + `Mediterranean` to isolate resolution cues from morphology.
- [ ] Fetch `fjordfish` and remove IMR Flodevigen mesocosm cod frames to keep strict wild-only North Atlantic coverage.

### C. Cross-Dataset Mixing & Baltic Sea Target Pipeline (RQ2)
- [ ] Create an RQ2 shared-species YAML config merging `ozfish`, `fishwio`, and `mediterranean` (`compute_rq2_mixing: true`, `filter_shared_species`) to inspect cross-dataset `top1_cross_dataset_accuracy` and `mean_normalized_dataset_entropy`.
- [ ] Add turbidity stress test comparing `brackish` / `fishinturbidwater` against clear-water reef datasets using high-D k-NN OOD scores.
- [ ] Fetch fish-active `hodor` sequences (`Kiel Fjord`) and integrate `schufi` frames once partner imagery is staged under `data/raw/schufi/`.
- [ ] Wire `scripts/crop_schufi_frames.py` (`grounding_dino` vs. `fishdet_m`) to crop unboxed Baltic monitoring frames cleanly.
- [ ] Verify out-of-sample `openTSNE` placement (`transform_new_points`) of held-out Baltic (`HODOR`/`SCHUFI`) crops onto the open-ocean reference map.

### D. Explorer & Thesis Reporting Polish
- [ ] Add multi-backbone comparison view to `explorer/index.html` (or generate per-backbone comparison table + summary scripts under `scripts/`).
- [ ] Add script to export LaTeX-formatted summary tables directly from `outputs/metrics/*/clustering_metrics.csv` and `projection_metrics.csv`.
- [ ] Prepare an anonymized open-dataset build of the interactive explorer (excluding partner-restricted `SCHUFI` images) for GitHub Pages preview if desired.

---

## 5. Open Questions and Decisions

1. **DINOv3 with higher dimensions (`ViT-L/16` `1024-D`, `ViT-H+/16` `1280-D`, `ViT-7B` `4096-D`): will local Apple M4 Pro work, or do we need Google / University GPU access?**
   - **What runs locally on Apple M4 Pro (via PyTorch `mps` Metal GPU):**
     Because all backbones run **frozen inference only** (no training or gradients) and embeddings are cached to `.npz` once per image set, `dinov3_vitb16` (`768-D`, `~86M` params), `dinov3_vitl16` (`1024-D`, `~300M` params), and curated subsets (`~1k-10k` crops) of `dinov3_vith16plus` (`1280-D`, `~840M` params, the top backbone in Markoff et al. 2026) can run locally in Terminal with PyTorch `mps` using moderate batch sizes (`batch_size: 16` or `32`). Note: run directly in native macOS Terminal so Metal `mps` is enabled (`torch.backends.mps.is_available() == True`).
   - **When external GPU (OVGU University Cluster or Google Colab Pro) becomes necessary:**
     - Extracting full multi-dataset corpora at scale (`50,000-200,000+` un-sampled crops across `OzFish`, `FishWIO`, `FRGT`, and `DeepFish`), OR
     - Evaluating ultra-large `DINOv3-7B` (`4096-D`, multi-GB weights needing `>= 16-24 GB` dedicated VRAM in fp16/bf16).
   - **Action to verify next:** Benchmark `dinov3_vitl16` and `dinov3_vith16plus` on `fishwio_dinov3_demo` (`900` images) natively in Terminal on M4 Pro (`mps`) to record runtime and peak memory, and simultaneously confirm OVGU / FIN compute cluster GPU access (e.g. A100/L40S/H100) vs. Google Colab Pro for full multi-dataset sweeps.

2. **Detector choice for unboxed Baltic frames (`SCHUFI` / `HODOR`):** Confirm with supervisors (`Grounding DINO` zero-shot prompt vs. `FishDet-M` supervised detector) before implementing `scripts/crop_schufi_frames.py`.

3. **Domain-specific vision-language backbones (`BioCLIP 2` vs. `AquaticCLIP`):** Confirm whether `AquaticCLIP` weights are publicly accessible and compatible with `open_clip` / `transformers` alongside `BioCLIP 2`.

4. **`SCHUFI` evaluation design (`RQ2`):** Confirm whether partial species annotations or expert biology spot-checks will be used to validate `SCHUFI` cross-dataset neighbor matches and OOD flags.

5. **Public Visual Atlas hosting:** Decide whether to publish anonymized open-dataset explorer outputs (`FRGT`, `FishWIO`, `OzFish`) on GitHub Pages while keeping partner `SCHUFI` imagery strictly local.
