"""Download and inspect wild fish datasets for local thesis runs.

Raw datasets are kept outside Git tracking under ``data/raw/<dataset>/``.
Public archives on Zenodo, Fish4Knowledge (FRGT), and Kaggle can be fetched
automatically with this script, while partner or multi-part repositories
(OzFish, DeepFish, FishInTurbidWater, HODOR, SCHUFI) include step-by-step
instructions and official links.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and download instructions for a dataset source."""

    key: str
    role: str
    region: str
    doi_or_url: str
    method: str  # zenodo | direct_tar | kaggle | hodor_python | manual
    source_id: str
    notes: str


DATASETS: dict[str, DatasetSpec] = {
    "frgt": DatasetSpec(
        key="frgt",
        role="RQ1/RQ2",
        region="Taiwan reefs",
        doi_or_url="https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/",
        method="direct_tar",
        source_id="https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar",
        notes="27,370 fish crops and 27,370 binary masks across 23 classes.",
    ),
    "ozfish": DatasetSpec(
        key="ozfish",
        role="RQ1/RQ2",
        region="Australia (BRUVS)",
        doi_or_url="https://doi.org/10.25845/5e28f062c5097",
        method="manual",
        source_id="https://github.com/open-AIMS/ozfish",
        notes="80,823 fish crops across 500+ taxa; use group-aware camera sampling.",
    ),
    "fishwio": DatasetSpec(
        key="fishwio",
        role="RQ1/RQ2",
        region="Mayotte (W. Indian Ocean)",
        doi_or_url="https://doi.org/10.5281/zenodo.18600403",
        method="zenodo",
        source_id="18600403",
        notes="114,687 wild reef crops across 124 species + 8 background classes.",
    ),
    "mediterranean": DatasetSpec(
        key="mediterranean",
        role="RQ1/RQ2",
        region="Mallorca (W. Mediterranean)",
        doi_or_url="https://doi.org/10.5281/zenodo.7534425",
        method="zenodo",
        source_id="7534425",
        notes="~1,590 coastal frames and ~18,500 YOLO boxes across ~20 species.",
    ),
    "fjordfish": DatasetSpec(
        key="fjordfish",
        role="RQ1/RQ2",
        region="Norway & Sweden coastal fjords",
        doi_or_url="https://doi.org/10.5281/zenodo.17950781",
        method="zenodo",
        source_id="17950781",
        notes="Filter out IMR Flodevigen mesocosm cod frames to keep wild-only scope.",
    ),
    "brackish": DatasetSpec(
        key="brackish",
        role="RQ2",
        region="Limfjorden (Denmark)",
        doi_or_url="https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset",
        method="kaggle",
        source_id="aalborguniversity/brackish-dataset",
        notes="15,084 temperate turbid frames with 6 coarse classes (no species IDs).",
    ),
    "deepfish": DatasetSpec(
        key="deepfish",
        role="RQ2",
        region="Tropical Australia",
        doi_or_url="https://github.com/alzayats/DeepFish",
        method="manual",
        source_id="https://github.com/alzayats/DeepFish",
        notes="~40k habitat frames without species labels; used for frame/habitat checks.",
    ),
    "fishinturbidwater": DatasetSpec(
        key="fishinturbidwater",
        role="Optional",
        region="Tropical Australia (turbid)",
        doi_or_url="https://doi.org/10.1016/j.ecoinf.2023.102303",
        method="manual",
        source_id="https://doi.org/10.1016/j.ecoinf.2023.102303",
        notes="51 turbid estuarine videos (~664 min) for visibility stress checks.",
    ),
    "hodor": DatasetSpec(
        key="hodor",
        role="RQ2",
        region="Kiel Fjord (Baltic Sea)",
        doi_or_url="https://doi.org/10.1594/PANGAEA.980000",
        method="hodor_python",
        source_id="10.1594/PANGAEA.980000",
        notes="German Baltic observatory; use hodor-python to fetch fish-active clips.",
    ),
    "schufi": DatasetSpec(
        key="schufi",
        role="Target",
        region="Baltic Sea (SCHUFI)",
        doi_or_url="University of Rostock / SCHUFI Project",
        method="manual",
        source_id="partner_access",
        notes="Partner Baltic monitoring imagery; place frames in data/raw/schufi/.",
    ),
}

AUTO_DATASETS = ("frgt", "fishwio", "fjordfish", "mediterranean", "brackish")


def is_present(dataset_dir: Path) -> bool:
    """Return True when the local dataset folder exists and contains files."""
    if not dataset_dir.is_dir():
        return False
    return any(p.name != ".gitkeep" for p in dataset_dir.iterdir())


def list_datasets(output_dir: Path) -> None:
    """Print local status and download method for every configured dataset."""
    header = f"{'Dataset':<18} {'Status':<10} {'Role':<9} {'Method':<13} {'Source / DOI'}"
    print(header)
    print("-" * len(header))
    for spec in DATASETS.values():
        status = "[LOCAL]" if is_present(output_dir / spec.key) else "[MISSING]"
        print(
            f"{spec.key:<18} {status:<10} {spec.role:<9} {spec.method:<13} {spec.doi_or_url}"
        )


def download_file(url: str, dest: Path) -> None:
    """Stream a remote file to disk without loading the full archive into RAM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fish-clustering/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)


def safe_extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract .zip or .tar* archives while blocking path-traversal entries."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target = (root / member.filename).resolve()
                if not str(target).startswith(str(root)):
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
            zf.extractall(root)
    elif name.endswith(".tar") or name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                target = (root / member.name).resolve()
                if not str(target).startswith(str(root)):
                    raise ValueError(f"Unsafe tar member path: {member.name}")
            tf.extractall(root)


def fetch_zenodo(
    spec: DatasetSpec,
    dest_dir: Path,
    dry_run: bool,
    extract: bool,
) -> None:
    """Download public files from a Zenodo record ID."""
    api_url = f"https://zenodo.org/api/records/{spec.source_id}"
    if dry_run:
        print(f"[DRY-RUN] {spec.key}: query {api_url} -> save under {dest_dir}")
        print(f"          Note: {spec.notes}")
        return

    req = urllib.request.Request(api_url, headers={"User-Agent": "fish-clustering/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        record = json.loads(resp.read().decode("utf-8"))

    files = record.get("files", [])
    if not files:
        raise RuntimeError(f"No downloadable files listed for Zenodo record {spec.source_id}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for file_info in files:
        fname = file_info["key"]
        file_url = file_info["links"]["self"]
        out_path = dest_dir / fname
        if not out_path.exists():
            print(f"Downloading {spec.key}: {fname} ...")
            download_file(file_url, out_path)
        if extract and out_path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
            print(f"Extracting {fname} -> {dest_dir / 'extracted'} ...")
            safe_extract_archive(out_path, dest_dir / "extracted")


def fetch_direct_tar(
    spec: DatasetSpec,
    dest_dir: Path,
    dry_run: bool,
    extract: bool,
) -> None:
    """Download and optionally extract a direct tar archive URL."""
    archive_name = Path(spec.source_id).name
    out_path = dest_dir / archive_name
    if dry_run:
        print(f"[DRY-RUN] {spec.key}: download {spec.source_id} -> {out_path}")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        print(f"Downloading {spec.key}: {archive_name} ...")
        download_file(spec.source_id, out_path)
    if extract:
        print(f"Extracting {archive_name} -> {dest_dir} ...")
        safe_extract_archive(out_path, dest_dir)


def fetch_kaggle(spec: DatasetSpec, dest_dir: Path, dry_run: bool, extract: bool) -> None:
    """Download a Kaggle dataset via the official Kaggle CLI."""
    cmd = ["kaggle", "datasets", "download", "-d", spec.source_id, "-p", str(dest_dir)]
    if extract:
        cmd.append("--unzip")
    if dry_run:
        print(f"[DRY-RUN] {spec.key}: {' '.join(cmd)}")
        return

    if shutil.which("kaggle") is None:
        print(
            f"[SKIP] {spec.key}: 'kaggle' CLI not found. Install 'kaggle', place "
            "~/.kaggle/kaggle.json, then run:\n  " + " ".join(cmd)
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


def print_manual_guide(spec: DatasetSpec, dest_dir: Path) -> None:
    """Print targeted instructions for PANGAEA HODOR or manual partner datasets."""
    if spec.method == "hodor_python":
        print(f"[GUIDE] {spec.key} ({spec.region}) -> place files in {dest_dir}")
        print(f"        Source: {spec.doi_or_url}")
        print(
            "        Use 'pip install hodor-python' to filter fish-active optical "
            "sequences before downloading instead of pulling all 400+ hours."
        )
        print(f"        Note: {spec.notes}")
        return

    print(f"[MANUAL] {spec.key} ({spec.region}) -> place files in {dest_dir}")
    print(f"         Official source: {spec.doi_or_url}")
    if spec.source_id not in {spec.doi_or_url, "partner_access"}:
        print(f"         Helper repo:     {spec.source_id}")
    print(f"         Note: {spec.notes}")


def fetch_dataset(
    key: str,
    output_dir: Path,
    dry_run: bool,
    extract: bool,
) -> None:
    """Dispatch dataset acquisition according to its repository type."""
    spec = DATASETS[key]
    dest_dir = output_dir / spec.key

    if not dry_run and is_present(dest_dir):
        print(f"[OK] {spec.key}: already present at {dest_dir}")
        return

    if spec.method == "zenodo":
        fetch_zenodo(spec, dest_dir, dry_run=dry_run, extract=extract)
    elif spec.method == "direct_tar":
        fetch_direct_tar(spec, dest_dir, dry_run=dry_run, extract=extract)
    elif spec.method == "kaggle":
        fetch_kaggle(spec, dest_dir, dry_run=dry_run, extract=extract)
    else:
        print_manual_guide(spec, dest_dir)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for listing or downloading datasets."""
    parser = argparse.ArgumentParser(
        description="Inspect and download wild fish datasets into data/raw/."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show local dataset status ([LOCAL] vs [MISSING]) and exit.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset key to download or inspect (repeatable).",
    )
    parser.add_argument(
        "--all-auto",
        action="store_true",
        help="Download all automated public datasets (frgt, fishwio, fjordfish, mediterranean, brackish).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target directory for raw datasets (default: data/raw).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned download actions without downloading files.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep downloaded zip/tar archives without unpacking them.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.list or (not args.dataset and not args.all_auto):
        list_datasets(args.output_dir)
        return

    keys = list(AUTO_DATASETS) if args.all_auto else list(dict.fromkeys(args.dataset or []))
    for key in keys:
        fetch_dataset(
            key=key,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            extract=not args.no_extract,
        )


if __name__ == "__main__":
    main()
