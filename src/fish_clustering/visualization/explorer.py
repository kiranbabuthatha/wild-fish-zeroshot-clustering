"""Interactive HTML/JS WebGL embedding explorer with thumbnails and high-D neighbors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, features
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from fish_clustering.data.preprocessing import load_image_sample


def _sanitize_json(obj: Any) -> Any:
    """Recursively convert numpy types and NaN/inf floats into strict JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_json(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if obj is None:
        return None
    return obj


def _resolve_raw_image_exists(raw_path: str, raw_root: Path | None) -> bool:
    if not raw_path or str(raw_path).strip() == "":
        return False
    candidate = Path(raw_path)
    if candidate.is_file():
        return True
    if raw_root is not None and (raw_root / candidate).is_file():
        return True
    return False


def export_sample_thumbnails(
    manifest: pd.DataFrame,
    thumbs_dir: str | Path,
    *,
    raw_root: Path | None = None,
    max_size: int = 192,
    quality: int = 82,
) -> tuple[list[str], list[str]]:
    """Export resized RGB (and optional FRGT masked) sample thumbnails for web viewing."""
    out_dir = Path(thumbs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_webp = bool(features.check("webp"))
    ext = "webp" if use_webp else "jpg"
    img_format = "WEBP" if use_webp else "JPEG"

    thumb_paths: list[str] = []
    mask_thumb_paths: list[str] = []

    for idx, rec in enumerate(manifest.to_dict(orient="records")):
        raw_img = str(rec.get("image_path") or "")
        if not _resolve_raw_image_exists(raw_img, raw_root):
            thumb_paths.append("")
            mask_thumb_paths.append("")
            continue

        img = load_image_sample(
            rec,
            raw_root=raw_root,
            use_foreground_mask=False,
            crop_from_bbox=True,
        )
        img.thumbnail((max_size, max_size), resample=Image.BICUBIC)
        thumb_name = f"{idx:05d}.{ext}"
        img.save(out_dir / thumb_name, format=img_format, quality=quality)
        thumb_paths.append(f"thumbs/{thumb_name}")

        raw_mask = str(rec.get("mask_path") or "")
        if _resolve_raw_image_exists(raw_mask, raw_root):
            masked = load_image_sample(
                rec,
                raw_root=raw_root,
                use_foreground_mask=True,
                crop_from_bbox=True,
            )
            masked.thumbnail((max_size, max_size), resample=Image.BICUBIC)
            mask_name = f"{idx:05d}_mask.{ext}"
            masked.save(out_dir / mask_name, format=img_format, quality=quality)
            mask_thumb_paths.append(f"thumbs/{mask_name}")
        else:
            mask_thumb_paths.append("")

    return thumb_paths, mask_thumb_paths


def compute_top_k_neighbors(
    embeddings: np.ndarray,
    *,
    k: int = 5,
    metric: str = "cosine",
) -> list[list[dict[str, Any]]]:
    """Return top-k nearest neighbors and distances in high-D space for every sample."""
    arr = np.asarray(embeddings, dtype=np.float32)
    n_samples = arr.shape[0]
    if n_samples == 0:
        return []
    if n_samples == 1 or k <= 0:
        return [[] for _ in range(n_samples)]

    effective_k = min(int(k), n_samples - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric).fit(arr)
    dists, indices = nn.kneighbors(return_distance=True)

    neighbors_out: list[list[dict[str, Any]]] = []
    for i in range(n_samples):
        row_nbrs: list[dict[str, Any]] = []
        for idx_j, dist_j in zip(indices[i], dists[i]):
            j = int(idx_j)
            if j == i:
                continue
            row_nbrs.append({"idx": j, "dist": round(float(dist_j), 4)})
            if len(row_nbrs) == effective_k:
                break
        neighbors_out.append(row_nbrs)
    return neighbors_out


def compute_cluster_geometry(
    coords: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute per-cluster centroids and 2-D convex hulls for overlay rendering."""
    xy = np.asarray(coords, dtype=np.float32)
    lbls = np.asarray(labels, dtype=np.int64)
    centroids: dict[str, list[float]] = {}
    hulls: dict[str, list[list[float]]] = {}

    unique_clusters = sorted({int(lbl) for lbl in lbls.tolist() if lbl >= 0})
    for cid in unique_clusters:
        pts = xy[lbls == cid]
        if len(pts) == 0:
            continue
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        centroids[str(cid)] = [round(cx, 5), round(cy, 5)]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                closed = np.vstack([verts, verts[:1]])
                hulls[str(cid)] = [[round(float(x), 5), round(float(y), 5)] for x, y in closed]
            except Exception:
                continue

    return {"centroids": centroids, "hulls": hulls}


def _build_explorer_html(experiment_name: str) -> str:
    """Return the self-contained HTML/CSS/JS frontend template."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{experiment_name} - Wild Fish Embedding Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="explorer_data.js"></script>
  <style>
    :root {{
      --bg: #f5f7fa;
      --card: #ffffff;
      --border: #dce3ec;
      --text: #162130;
      --muted: #5a687a;
      --accent: #185adb;
      --accent-soft: #eaf1ff;
      --noise-bg: #fff3f5;
      --noise-border: #f3c4cc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }}
    .page {{
      max-width: 1520px;
      margin: 0 auto;
      padding: 20px 24px 48px;
    }}
    .header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-end;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 1.5rem;
      font-weight: 700;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 0.92rem;
      margin-top: 4px;
    }}
    .badge {{
      display: inline-block;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 600;
      font-size: 0.82rem;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #c9dbff;
    }}
    .controls-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: flex-end;
    }}
    .control-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 140px;
    }}
    .control-group label {{
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .control-group select, .control-group input[type="text"] {{
      padding: 7px 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font-size: 0.9rem;
    }}
    .control-search {{
      flex: 1 1 220px;
    }}
    .toggle-row {{
      display: flex;
      gap: 14px;
      align-items: center;
      padding-bottom: 6px;
      font-size: 0.88rem;
    }}
    .toggle-row label {{
      display: flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
    }}
    .btn {{
      padding: 7px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn:hover {{ background: #f0f4fa; }}
    .btn-primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .btn-primary:hover {{ background: #1248b8; }}
    .main-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    @media (max-width: 1080px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
    }}
    .plot-card, .inspector-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }}
    #embedding-plot {{
      width: 100%;
      height: 680px;
    }}
    .inspector-card {{
      position: sticky;
      top: 14px;
    }}
    .inspector-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }}
    .inspector-header h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    .thumb-pair {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .thumb-box {{
      flex: 1;
      background: #f0f3f8;
      border: 1px solid var(--border);
      border-radius: 8px;
      min-height: 150px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 6px;
      overflow: hidden;
    }}
    .thumb-box img {{
      max-width: 100%;
      max-height: 170px;
      object-fit: contain;
      border-radius: 4px;
    }}
    .thumb-caption {{
      font-size: 0.72rem;
      color: var(--muted);
      margin-top: 4px;
    }}
    .meta-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      margin-bottom: 12px;
    }}
    .meta-table td {{
      padding: 4px 0;
      border-bottom: 1px solid #eef2f7;
    }}
    .meta-table td:first-child {{
      color: var(--muted);
      width: 42%;
      font-weight: 500;
    }}
    .nbr-title {{
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      margin: 8px 0 6px;
    }}
    .nbr-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
    }}
    .nbr-item {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px;
      text-align: center;
      cursor: pointer;
      background: #fafbfd;
      font-size: 0.7rem;
    }}
    .nbr-item:hover {{ border-color: var(--accent); }}
    .nbr-item img {{
      width: 100%;
      height: 46px;
      object-fit: cover;
      border-radius: 4px;
      display: block;
      margin-bottom: 3px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
      gap: 10px;
      margin-bottom: 20px;
    }}
    .kpi {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
    }}
    .kpi-label {{
      font-size: 0.74rem;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 600;
    }}
    .kpi-val {{
      font-size: 1.2rem;
      font-weight: 700;
      margin-top: 2px;
    }}
    .section-title {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 10px 0 10px;
    }}
    .section-title h2 {{
      margin: 0;
      font-size: 1.15rem;
    }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 12px;
    }}
    .cluster-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      cursor: pointer;
      transition: border-color 0.15s ease, transform 0.12s ease;
    }}
    .cluster-card:hover {{
      border-color: var(--accent);
      transform: translateY(-1px);
    }}
    .cluster-card.noise {{
      background: var(--noise-bg);
      border-color: var(--noise-border);
    }}
    .cluster-card.active-filter {{
      border: 2px solid var(--accent);
      background: #f5f9ff;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 4px;
    }}
    .card-title {{ font-weight: 700; font-size: 0.96rem; }}
    .card-count {{ font-size: 0.8rem; color: var(--muted); }}
    .card-sub {{
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 8px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .card-thumbs {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 8px;
    }}
    .card-thumbs img {{
      width: 100%;
      height: 62px;
      object-fit: cover;
      border-radius: 6px;
      background: #edf1f7;
    }}
    .modal-backdrop {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(16, 24, 40, 0.72);
      z-index: 1000;
      padding: 28px;
      overflow-y: auto;
    }}
    .modal-backdrop.open {{ display: block; }}
    .modal-box {{
      max-width: 1120px;
      margin: 0 auto;
      background: var(--card);
      border-radius: 12px;
      padding: 18px 20px;
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 14px;
    }}
    .modal-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 10px;
      max-height: 70vh;
      overflow-y: auto;
    }}
    .modal-item {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px;
      font-size: 0.76rem;
    }}
    .modal-item img {{
      width: 100%;
      height: 110px;
      object-fit: cover;
      border-radius: 4px;
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <h1 id="exp-title">Wild Fish Embedding Explorer</h1>
      <div class="subtitle" id="exp-subtitle">Loading experiment metadata...</div>
    </div>
    <span class="badge" id="exp-badge">Zero-Shot Atlas</span>
  </div>

  <div class="controls-card">
    <div class="control-group">
      <label for="seed-select">Seed</label>
      <select id="seed-select"></select>
    </div>
    <div class="control-group">
      <label for="reduction-select">Reduction</label>
      <select id="reduction-select"></select>
    </div>
    <div class="control-group">
      <label for="clustering-select">Clustering</label>
      <select id="clustering-select"></select>
    </div>
    <div class="control-group">
      <label for="color-select">Color By</label>
      <select id="color-select">
        <option value="cluster">Predicted Clusters</option>
        <option value="label">Ground Truth / Label</option>
        <option value="dataset">Dataset Source</option>
        <option value="distortion">Projection Distortion</option>
        <option value="ood">High-D k-NN OOD Score</option>
      </select>
    </div>
    <div class="control-group control-search">
      <label for="search-input">Search Species / ID / Dataset</label>
      <input id="search-input" type="text" placeholder="Type e.g. cluster_3, Caranx, frgt..." />
    </div>
    <div class="toggle-row">
      <label><input type="checkbox" id="show-centroids" checked /> Centroids</label>
      <label><input type="checkbox" id="show-hulls" /> Hulls</label>
      <button class="btn" id="clear-filter-btn" type="button">Reset Filter</button>
    </div>
  </div>

  <div class="main-grid">
    <div class="plot-card">
      <div id="embedding-plot"></div>
    </div>
    <div class="inspector-card">
      <div class="inspector-header">
        <h3>Sample Inspector</h3>
        <span class="badge" id="pin-status">Hover or Click Point</span>
      </div>
      <div class="thumb-pair">
        <div class="thumb-box" id="inspector-img-box">
          <span class="thumb-caption">No sample selected</span>
        </div>
        <div class="thumb-box" id="inspector-mask-box" style="display:none;">
          <span class="thumb-caption">Foreground Mask</span>
        </div>
      </div>
      <table class="meta-table" id="inspector-meta"></table>
      <div class="nbr-title">Top-5 High-D Nearest Neighbors</div>
      <div class="nbr-grid" id="inspector-nbrs"></div>
    </div>
  </div>

  <div class="kpi-grid" id="kpi-grid"></div>

  <div class="section-title">
    <h2>Cluster Summary Cards</h2>
    <span class="subtitle">Click Highlight to filter the scatter plot, or click a card to open its full gallery.</span>
  </div>
  <div class="cards-grid" id="cards-grid"></div>
</div>

<div class="modal-backdrop" id="cluster-modal">
  <div class="modal-box">
    <div class="modal-header">
      <h3 id="modal-title" style="margin:0;">Cluster Gallery</h3>
      <button class="btn" id="modal-close-btn" type="button">Close</button>
    </div>
    <div class="modal-grid" id="modal-grid"></div>
  </div>
</div>

<script>
(function() {{
  const PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
    "#e377c2","#17becf","#bcbd22","#393b79","#637939","#8c6d31",
    "#843c39","#7b4173","#3182bd","#e6550d","#31a354","#756bb1"
  ];

  let DATA = window.FISH_ATLAS_DATA || null;
  const state = {{
    seed: null,
    reduction: null,
    clustering: null,
    colorBy: "cluster",
    search: "",
    highlightCluster: null,
    pinnedIdx: null,
    showCentroids: true,
    showHulls: false
  }};

  function fmt(v, digits = 3) {{
    if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
    return typeof v === "number" ? v.toFixed(digits) : String(v);
  }}

  function displayLabel(sample) {{
    return sample.species || sample.label_raw || "unlabeled";
  }}

  function populateSelectors() {{
    document.getElementById("exp-title").textContent = `${{DATA.experiment}} - Visual Atlas`;
    document.getElementById("exp-subtitle").textContent =
      `Backbone: ${{DATA.backbone}} (${{DATA.pooling}}) | Samples: ${{DATA.samples.length}} | Embedding Dim: ${{DATA.embedding_dim}}`;
    document.getElementById("exp-badge").textContent = `${{DATA.backbone}}`;

    const seedSel = document.getElementById("seed-select");
    seedSel.innerHTML = "";
    DATA.seeds.forEach((s) => {{
      const opt = document.createElement("option");
      opt.value = String(s);
      opt.textContent = String(s);
      seedSel.appendChild(opt);
    }});
    state.seed = String(DATA.seeds[0]);
    seedSel.value = state.seed;

    refreshReductionOptions();
  }}

  function refreshReductionOptions() {{
    const redSel = document.getElementById("reduction-select");
    const reds = Object.keys(DATA.runs[state.seed]);
    redSel.innerHTML = "";
    reds.forEach((r) => {{
      const opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r.toUpperCase();
      redSel.appendChild(opt);
    }});
    if (!reds.includes(state.reduction)) {{
      state.reduction = reds.includes("umap") ? "umap" : (reds.includes("tsne") ? "tsne" : reds[0]);
    }}
    redSel.value = state.reduction;
    refreshClusteringOptions();
  }}

  function refreshClusteringOptions() {{
    const clSel = document.getElementById("clustering-select");
    const run = DATA.runs[state.seed][state.reduction];
    const methods = Object.keys(run.clusterings);
    clSel.innerHTML = "";
    methods.forEach((m) => {{
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m.toUpperCase();
      clSel.appendChild(opt);
    }});
    if (!methods.includes(state.clustering)) {{
      state.clustering = methods[0];
    }}
    clSel.value = state.clustering;
  }}

  function currentRun() {{
    return DATA.runs[state.seed][state.reduction];
  }}

  function currentClustering() {{
    return currentRun().clusterings[state.clustering];
  }}

  function matchesFilter(idx, labels) {{
    const q = state.search.trim().toLowerCase();
    const s = DATA.samples[idx];
    const textOk = !q ||
      s.image_id.toLowerCase().includes(q) ||
      s.dataset.toLowerCase().includes(q) ||
      displayLabel(s).toLowerCase().includes(q);
    const clOk = state.highlightCluster === null || String(labels[idx]) === state.highlightCluster;
    return textOk && clOk;
  }}

  function renderPlot() {{
    const run = currentRun();
    const cl = currentClustering();
    const coords = run.coords;
    const labels = cl.labels;
    const n = DATA.samples.length;
    const traces = [];

    const hoverTexts = DATA.samples.map((s, i) => {{
      const clName = labels[i] === -1 ? "Noise (-1)" : `Cluster ${{labels[i]}}`;
      const distVal = run.distortion ? fmt(run.distortion[i]) : "n/a";
      return `${{s.image_id}}<br>Dataset: ${{s.dataset}}<br>Label: ${{displayLabel(s)}}<br>${{clName}}<br>Distortion: ${{distVal}} | OOD: ${{fmt(s.ood_score)}}`;
    }});

    if (state.colorBy === "distortion" || state.colorBy === "ood") {{
      const vals = state.colorBy === "distortion"
        ? (run.distortion || new Array(n).fill(0))
        : DATA.samples.map((s) => s.ood_score ?? 0);
      const opacities = DATA.samples.map((_, i) => matchesFilter(i, labels) ? 0.9 : 0.12);
      traces.push({{
        type: "scattergl",
        mode: "markers",
        x: coords.map((p) => p[0]),
        y: coords.map((p) => p[1]),
        customdata: DATA.samples.map((_, i) => i),
        text: hoverTexts,
        hoverinfo: "text",
        showlegend: false,
        marker: {{
          size: 7,
          opacity: opacities,
          color: vals,
          colorscale: state.colorBy === "distortion" ? "Turbo" : "Plasma",
          showscale: true,
          colorbar: {{ title: state.colorBy === "distortion" ? "Distortion" : "OOD", thickness: 14 }}
        }}
      }});
    }} else {{
      const groups = new Map();
      for (let i = 0; i < n; i++) {{
        let key;
        if (state.colorBy === "cluster") {{
          key = labels[i] === -1 ? "Noise (-1)" : `Cluster ${{labels[i]}}`;
        }} else if (state.colorBy === "dataset") {{
          key = DATA.samples[i].dataset || "unknown";
        }} else {{
          key = displayLabel(DATA.samples[i]);
        }}
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
      }}

      const sortedKeys = Array.from(groups.keys()).sort((a, b) => {{
        if (a === "Noise (-1)") return -1;
        if (b === "Noise (-1)") return 1;
        const na = a.startsWith("Cluster ") ? Number(a.slice(8)) : NaN;
        const nb = b.startsWith("Cluster ") ? Number(b.slice(8)) : NaN;
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
        return a.localeCompare(b);
      }});

      sortedKeys.forEach((key, gIdx) => {{
        const idxs = groups.get(key);
        const isNoise = key === "Noise (-1)";
        const color = isNoise ? "#9aa5b1" : PALETTE[gIdx % PALETTE.length];
        traces.push({{
          type: "scattergl",
          mode: "markers",
          name: `${{key}} (${{idxs.length}})`,
          x: idxs.map((i) => coords[i][0]),
          y: idxs.map((i) => coords[i][1]),
          customdata: idxs,
          text: idxs.map((i) => hoverTexts[i]),
          hoverinfo: "text",
          marker: {{
            size: isNoise ? 5 : 7,
            color: color,
            opacity: idxs.map((i) => matchesFilter(i, labels) ? (isNoise ? 0.55 : 0.88) : 0.1),
            line: {{ width: 0.4, color: "#1d2633" }}
          }}
        }});
      }});

      if (state.colorBy === "cluster" && state.showHulls && cl.geometry && cl.geometry.hulls) {{
        Object.entries(cl.geometry.hulls).forEach(([cid, poly]) => {{
          traces.push({{
            type: "scatter",
            mode: "lines",
            x: poly.map((p) => p[0]),
            y: poly.map((p) => p[1]),
            line: {{ width: 1.3, dash: "dot", color: "#334155" }},
            hoverinfo: "skip",
            showlegend: false
          }});
        }});
      }}

      if (state.colorBy === "cluster" && state.showCentroids && cl.geometry && cl.geometry.centroids) {{
        const cIds = Object.keys(cl.geometry.centroids);
        if (cIds.length > 0) {{
          traces.push({{
            type: "scatter",
            mode: "markers+text",
            name: "Centroids",
            x: cIds.map((id) => cl.geometry.centroids[id][0]),
            y: cIds.map((id) => cl.geometry.centroids[id][1]),
            text: cIds.map((id) => id),
            textposition: "top center",
            hoverinfo: "text",
            marker: {{ symbol: "x", size: 10, color: "#0f172a", line: {{ width: 1.5, color: "#ffffff" }} }}
          }});
        }}
      }}
    }}

    const layout = {{
      title: {{ text: `${{state.reduction.toUpperCase()}} - ${{state.clustering.toUpperCase()}} (${{state.colorBy}})`, x: 0.02 }},
      template: "plotly_white",
      margin: {{ t: 48, l: 50, r: 24, b: 96 }},
      hovermode: "closest",
      xaxis: {{ title: `${{state.reduction.toUpperCase()}} 1`, zeroline: false }},
      yaxis: {{ title: `${{state.reduction.toUpperCase()}} 2`, zeroline: false }},
      legend: {{ orientation: "h", y: -0.18, x: 0, font: {{ size: 11 }} }}
    }};

    Plotly.react("embedding-plot", traces, layout, {{ responsive: true }});
    const plotEl = document.getElementById("embedding-plot");
    plotEl.removeAllListeners && plotEl.removeAllListeners("plotly_hover");
    plotEl.removeAllListeners && plotEl.removeAllListeners("plotly_click");

    plotEl.on("plotly_hover", (ev) => {{
      if (state.pinnedIdx !== null) return;
      const pt = ev.points && ev.points[0];
      if (pt && pt.customdata !== undefined) showSample(Number(pt.customdata), false);
    }});

    plotEl.on("plotly_click", (ev) => {{
      const pt = ev.points && ev.points[0];
      if (pt && pt.customdata !== undefined) {{
        const idx = Number(pt.customdata);
        state.pinnedIdx = state.pinnedIdx === idx ? null : idx;
        showSample(idx, state.pinnedIdx !== null);
      }}
    }});
  }}

  function showSample(idx, isPinned) {{
    const s = DATA.samples[idx];
    const cl = currentClustering();
    const run = currentRun();
    document.getElementById("pin-status").textContent = isPinned ? `Pinned #${{idx}} (click again to unpin)` : `Hover #${{idx}}`;

    const imgBox = document.getElementById("inspector-img-box");
    imgBox.innerHTML = s.thumb
      ? `<img src="${{s.thumb}}" alt="${{s.image_id}}" /><span class="thumb-caption">Original Crop</span>`
      : `<span class="thumb-caption">No thumbnail (${{s.image_id}})</span>`;

    const maskBox = document.getElementById("inspector-mask-box");
    if (s.mask_thumb) {{
      maskBox.style.display = "flex";
      maskBox.innerHTML = `<img src="${{s.mask_thumb}}" alt="mask" /><span class="thumb-caption">Masked Foreground</span>`;
    }} else {{
      maskBox.style.display = "none";
    }}

    const lbl = cl.labels[idx] === -1 ? "Noise (-1)" : `Cluster ${{cl.labels[idx]}}`;
    const distVal = run.distortion ? fmt(run.distortion[idx]) : "n/a";
    document.getElementById("inspector-meta").innerHTML = `
      <tr><td>Image ID</td><td>${{s.image_id}}</td></tr>
      <tr><td>Dataset</td><td>${{s.dataset}} (${{s.source_type || "wild"}})</td></tr>
      <tr><td>Label</td><td>${{displayLabel(s)}}</td></tr>
      <tr><td>Assigned Cluster</td><td>${{lbl}}</td></tr>
      <tr><td>Distortion / OOD</td><td>${{distVal}} / ${{fmt(s.ood_score)}}</td></tr>
      <tr><td>Video ID</td><td>${{s.video_id || "n/a"}}</td></tr>
    `;

    const nbrEl = document.getElementById("inspector-nbrs");
    nbrEl.innerHTML = "";
    (s.neighbors || []).forEach((nb) => {{
      const ns = DATA.samples[nb.idx];
      const item = document.createElement("div");
      item.className = "nbr-item";
      item.title = `${{ns.image_id}} | ${{ns.dataset}} | ${{displayLabel(ns)}} (d=${{nb.dist}})`;
      item.innerHTML = `
        ${{ns.thumb ? `<img src="${{ns.thumb}}" alt="${{ns.image_id}}" />` : ""}}
        <div><strong>${{ns.dataset}}</strong></div>
        <div>d=${{nb.dist.toFixed(2)}}</div>
      `;
      item.addEventListener("click", () => {{
        state.pinnedIdx = nb.idx;
        showSample(nb.idx, true);
      }});
      nbrEl.appendChild(item);
    }});
  }}

  function renderKPIs() {{
    const run = currentRun();
    const cl = currentClustering();
    const m = cl.metrics || {{}};
    const pq = run.projection_metrics || {{}};
    const rq2 = DATA.rq2 || {{}};

    const kpis = [
      ["Samples", DATA.samples.length],
      ["Clusters", m.n_clusters ?? "n/a"],
      ["Noise", m.n_noise ?? 0],
      ["V-Measure", fmt(m.v_measure)],
      ["ARI", fmt(m.ari)],
      ["AMI", fmt(m.ami)],
      ["Silhouette", fmt(m.silhouette)],
      ["DBCV", fmt(m.dbcv)],
      ["Trustworthiness", fmt(pq.trustworthiness)],
      ["Continuity", fmt(pq.continuity)]
    ];
    if (rq2.mean_normalized_dataset_entropy !== undefined) {{
      kpis.push(["Dataset Mixing H", fmt(rq2.mean_normalized_dataset_entropy)]);
    }}

    document.getElementById("kpi-grid").innerHTML = kpis
      .map(([k, v]) => `<div class="kpi"><div class="kpi-label">${{k}}</div><div class="kpi-val">${{v}}</div></div>`)
      .join("");
  }}

  function renderClusterCards() {{
    const cl = currentClustering();
    const labels = cl.labels;
    const byCluster = new Map();
    labels.forEach((lbl, idx) => {{
      const k = String(lbl);
      if (!byCluster.has(k)) byCluster.set(k, []);
      byCluster.get(k).push(idx);
    }});

    const sortedCids = Array.from(byCluster.keys()).sort((a, b) => Number(a) - Number(b));
    const grid = document.getElementById("cards-grid");
    grid.innerHTML = "";

    sortedCids.forEach((cid) => {{
      const idxs = byCluster.get(cid);
      const isNoise = cid === "-1";
      const labelCounts = new Map();
      const dsCounts = new Map();
      idxs.forEach((i) => {{
        const lbl = displayLabel(DATA.samples[i]);
        const ds = DATA.samples[i].dataset;
        labelCounts.set(lbl, (labelCounts.get(lbl) || 0) + 1);
        dsCounts.set(ds, (dsCounts.get(ds) || 0) + 1);
      }});
      const topLbl = Array.from(labelCounts.entries()).sort((a, b) => b[1] - a[1])[0];
      const purity = topLbl ? ((100 * topLbl[1]) / idxs.length).toFixed(1) : "0.0";
      const dsSummary = Array.from(dsCounts.entries()).map(([d, c]) => `${{d}}:${{c}}`).join(", ");

      const card = document.createElement("div");
      card.className = `cluster-card ${{isNoise ? "noise" : ""}} ${{state.highlightCluster === cid ? "active-filter" : ""}}`;
      const sampleThumbs = idxs.slice(0, 3).map((i) => {{
        const t = DATA.samples[i].thumb;
        return t ? `<img src="${{t}}" alt="${{DATA.samples[i].image_id}}" />` : `<div></div>`;
      }}).join("");

      card.innerHTML = `
        <div class="card-top">
          <span class="card-title">${{isNoise ? "Noise (-1)" : `Cluster ${{cid}}`}}</span>
          <span class="card-count">${{idxs.length}} images</span>
        </div>
        <div class="card-sub">${{topLbl ? `${{topLbl[0]}} (${{purity}}%)` : "unlabeled"}} | ${{dsSummary}}</div>
        <div class="card-thumbs">${{sampleThumbs}}</div>
        <button class="btn highlight-btn" type="button">${{state.highlightCluster === cid ? "Clear Highlight" : "Highlight on Map"}}</button>
      `;

      card.querySelector(".highlight-btn").addEventListener("click", (ev) => {{
        ev.stopPropagation();
        state.highlightCluster = state.highlightCluster === cid ? null : cid;
        renderPlot();
        renderClusterCards();
      }});

      card.addEventListener("click", () => openModal(cid, idxs));
      grid.appendChild(card);
    }});
  }}

  function openModal(cid, idxs) {{
    document.getElementById("modal-title").textContent =
      `${{cid === "-1" ? "Noise (-1)" : `Cluster ${{cid}}`}} (${{idxs.length}} images)`;
    const grid = document.getElementById("modal-grid");
    grid.innerHTML = idxs.slice(0, 200).map((i) => {{
      const s = DATA.samples[i];
      return `
        <div class="modal-item">
          ${{s.thumb ? `<img src="${{s.thumb}}" alt="${{s.image_id}}" />` : ""}}
          <div><strong>${{displayLabel(s)}}</strong></div>
          <div style="color:#5a687a;">${{s.dataset}} | ${{s.image_id}}</div>
        </div>
      `;
    }}).join("");
    document.getElementById("cluster-modal").classList.add("open");
  }}

  function bindEvents() {{
    document.getElementById("seed-select").addEventListener("change", (e) => {{
      state.seed = e.target.value;
      state.highlightCluster = null;
      refreshReductionOptions();
      renderAll();
    }});
    document.getElementById("reduction-select").addEventListener("change", (e) => {{
      state.reduction = e.target.value;
      state.highlightCluster = null;
      refreshClusteringOptions();
      renderAll();
    }});
    document.getElementById("clustering-select").addEventListener("change", (e) => {{
      state.clustering = e.target.value;
      state.highlightCluster = null;
      renderAll();
    }});
    document.getElementById("color-select").addEventListener("change", (e) => {{
      state.colorBy = e.target.value;
      renderPlot();
    }});
    document.getElementById("search-input").addEventListener("input", (e) => {{
      state.search = e.target.value;
      renderPlot();
    }});
    document.getElementById("show-centroids").addEventListener("change", (e) => {{
      state.showCentroids = e.target.checked;
      renderPlot();
    }});
    document.getElementById("show-hulls").addEventListener("change", (e) => {{
      state.showHulls = e.target.checked;
      renderPlot();
    }});
    document.getElementById("clear-filter-btn").addEventListener("click", () => {{
      state.search = "";
      state.highlightCluster = null;
      state.pinnedIdx = null;
      document.getElementById("search-input").value = "";
      renderPlot();
      renderClusterCards();
    }});
    document.getElementById("modal-close-btn").addEventListener("click", () => {{
      document.getElementById("cluster-modal").classList.remove("open");
    }});
    document.getElementById("cluster-modal").addEventListener("click", (e) => {{
      if (e.target.id === "cluster-modal") {{
        document.getElementById("cluster-modal").classList.remove("open");
      }}
    }});
  }}

  function renderAll() {{
    renderPlot();
    renderKPIs();
    renderClusterCards();
    if (DATA.samples.length > 0) {{
      showSample(state.pinnedIdx !== null ? state.pinnedIdx : 0, state.pinnedIdx !== null);
    }}
  }}

  async function init() {{
    if (!DATA) {{
      const resp = await fetch("explorer_data.json");
      DATA = await resp.json();
    }}
    populateSelectors();
    bindEvents();
    renderAll();
  }}

  init();
}})();
</script>
</body>
</html>
"""


def export_interactive_explorer(
    *,
    experiment_name: str,
    backbone: str,
    pooling: str,
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    run_data: dict[int, dict[str, dict[str, Any]]],
    ood_scores: np.ndarray,
    output_dir: str | Path,
    raw_root: Path | None = None,
    rq2_summary: dict[str, Any] | None = None,
    top_k_neighbors: int = 5,
    thumb_max_size: int = 192,
) -> Path:
    """Export interactive HTML/JS embedding explorer with WebP crops and high-D neighbors."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thumb_paths, mask_thumb_paths = export_sample_thumbnails(
        manifest,
        out_dir / "thumbs",
        raw_root=raw_root,
        max_size=thumb_max_size,
    )
    neighbors = compute_top_k_neighbors(embeddings, k=top_k_neighbors, metric="cosine")

    samples: list[dict[str, Any]] = []
    for i, rec in enumerate(manifest.to_dict(orient="records")):
        samples.append(
            {
                "image_id": str(rec.get("image_id", i)),
                "dataset": str(rec.get("dataset", "")),
                "source_type": str(rec.get("source_type", "")),
                "label_raw": str(rec.get("label_raw", "")),
                "species": str(rec.get("species", "")),
                "video_id": str(rec.get("video_id", "")),
                "ood_score": round(float(ood_scores[i]), 5) if i < len(ood_scores) else None,
                "thumb": thumb_paths[i],
                "mask_thumb": mask_thumb_paths[i],
                "neighbors": neighbors[i] if i < len(neighbors) else [],
            }
        )

    serialized_runs: dict[str, Any] = {}
    ordered_seeds = sorted(run_data.keys())
    for seed in ordered_seeds:
        red_dict: dict[str, Any] = {}
        for red_name, entry in run_data[seed].items():
            coords = np.asarray(entry["coords"], dtype=np.float32)
            distortion = entry.get("distortion")
            cl_serialized: dict[str, Any] = {}
            for method, cl_info in entry["clusterings"].items():
                lbls = np.asarray(cl_info["labels"], dtype=np.int64)
                cl_serialized[method] = {
                    "labels": lbls.tolist(),
                    "metrics": _sanitize_json(cl_info.get("metrics", {})),
                    "geometry": compute_cluster_geometry(coords, lbls),
                }
            red_dict[red_name] = {
                "coords": [[round(float(x), 5), round(float(y), 5)] for x, y in coords],
                "distortion": (
                    [round(float(d), 5) for d in distortion] if distortion is not None else None
                ),
                "projection_metrics": _sanitize_json(entry.get("projection_metrics", {})),
                "clusterings": cl_serialized,
            }
        serialized_runs[str(seed)] = red_dict

    payload = _sanitize_json(
        {
            "experiment": experiment_name,
            "backbone": backbone,
            "pooling": pooling,
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "seeds": ordered_seeds,
            "rq2": rq2_summary or {},
            "samples": samples,
            "runs": serialized_runs,
        }
    )

    json_text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    (out_dir / "explorer_data.json").write_text(json_text, encoding="utf-8")
    (out_dir / "explorer_data.js").write_text(
        f"window.FISH_ATLAS_DATA = {json_text};\n", encoding="utf-8"
    )
    html_path = out_dir / "index.html"
    html_path.write_text(_build_explorer_html(experiment_name), encoding="utf-8")
    return html_path
