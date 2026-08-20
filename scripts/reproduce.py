#!/usr/bin/env python3
"""Bootstrap the environment, public data, tests, and first real-data notebook."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "interleavedVerbalBehaviors.zip"
ARCHIVE_URL = (
    "https://github.com/ronnuriel/NeuralSpeechGeometry/"
    "releases/download/data-v1/interleavedVerbalBehaviors.zip"
)
ARCHIVE_SIZE = 777_419_058
ARCHIVE_SHA256 = "19f90f09f2ea32f1428b7cc1c7dd8c0606dfbc988c57fcaf64aea03e77b9d409"
ARCHIVE_PATH = ROOT / "data" / "raw" / "downloads" / ARCHIVE_NAME
EXTRACT_ROOT = ROOT / "data" / "raw"
EXPECTED_EXTRACTED_FILES = {
    "interleavedVerbalBehaviors/t12.2024.04.11_interleavedVerbalBehaviors_raw.mat": 4_830_190,
    "interleavedVerbalBehaviors/t15.2024.06.14_interleavedVerbalBehaviors_raw.mat": 356_757_442,
    "interleavedVerbalBehaviors/t16.2024.07.17_interleavedVerbalBehaviors_raw.mat": 126_928_854,
    "interleavedVerbalBehaviors/t17.2024.12.09_interleavedVerbalBehaviors_raw.mat": 292_407_710,
    "interleavedVerbalBehaviors/interleavedVerbalBehaviors_readme.txt": 2_669,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a potentially large file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _human_size(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def verify_archive(path: Path) -> None:
    """Verify size, checksum, and ZIP integrity against the Dryad manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != ARCHIVE_SIZE:
        raise RuntimeError(
            f"Unexpected archive size: {actual_size}; expected {ARCHIVE_SIZE}. "
            f"Move the invalid file aside and retry: {path}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}. Move the invalid file aside and retry."
        )
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"ZIP integrity failed at member: {bad_member}")


def download_archive(url: str, destination: Path) -> None:
    """Download with HTTP Range resume support and verify before finalizing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Data archive already exists: {destination}")
        verify_archive(destination)
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "NeuralSpeechGeometry-reproducer/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"Resuming download at {_human_size(existing)}")
    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed/explicit URL
        status = getattr(response, "status", None) or response.getcode()
        append = bool(existing and status == 206)
        if existing and not append:
            print("Server did not honor Range; restarting the partial download.")
            existing = 0
        mode = "ab" if append else "wb"
        downloaded = existing
        last_report = time.monotonic()
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 5:
                    percent = 100 * downloaded / ARCHIVE_SIZE
                    print(
                        f"Downloaded {_human_size(downloaded)} / "
                        f"{_human_size(ARCHIVE_SIZE)} ({percent:.1f}%)",
                        flush=True,
                    )
                    last_report = time.monotonic()

    print("Verifying downloaded archive...")
    verify_archive(partial)
    partial.replace(destination)
    print(f"Verified data archive: {destination}")


def _extracted_data_complete() -> bool:
    return all(
        (EXTRACT_ROOT / relative_path).is_file()
        and (EXTRACT_ROOT / relative_path).stat().st_size == expected_size
        for relative_path, expected_size in EXPECTED_EXTRACTED_FILES.items()
    )


def extract_archive(path: Path) -> None:
    """Safely extract the verified archive and check the expected file inventory."""
    if _extracted_data_complete():
        print("Extracted interleaved data already match the manifest.")
        return
    with zipfile.ZipFile(path) as archive:
        root = EXTRACT_ROOT.resolve()
        for member in archive.infolist():
            target = (EXTRACT_ROOT / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(EXTRACT_ROOT)
    if not _extracted_data_complete():
        raise RuntimeError("Extraction finished, but the expected data inventory is incomplete")
    print(f"Extracted and verified data under: {EXTRACT_ROOT}")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def prepare_environment(venv_dir: Path) -> Path:
    """Create an isolated environment and install the project plus test tools."""
    python = _venv_python(venv_dir)
    if not python.is_file():
        print(f"Creating virtual environment: {venv_dir}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    print("Installing project dependencies...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "-e", f"{ROOT}[dev]"],
        cwd=ROOT,
        check=True,
    )
    return python


def run_checks_and_analysis(python: Path, *, skip_tests: bool) -> Path:
    """Run tests and execute the first real-data notebook reproducibly."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    if not skip_tests:
        subprocess.run([str(python), "-m", "pytest"], cwd=ROOT, env=env, check=True)

    output_dir = ROOT / "results" / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ROOT / "configs" / "t15_interleaved_binnedtx.yaml"
    env["KUNZ_CONFIG"] = str(config)
    subprocess.run(
        [
            str(python),
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(ROOT / "notebooks" / "02_t15_real_data_executed.ipynb"),
            "--output",
            "t15_interleaved_binnedtx_executed.ipynb",
            "--output-dir",
            str(output_dir),
            "--ExecutePreprocessor.timeout=600",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return output_dir / "t15_interleaved_binnedtx_executed.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, verify, extract, test, and run the first real-data analysis"
    )
    parser.add_argument(
        "--archive-url",
        default=os.environ.get("KUNZ_DATA_URL", ARCHIVE_URL),
        help="Override the byte-identical data mirror URL",
    )
    parser.add_argument("--data-only", action="store_true", help="Stop after data extraction")
    parser.add_argument("--analysis-only", action="store_true", help="Require existing data")
    parser.add_argument("--skip-install", action="store_true", help="Use the current Python")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run pytest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_only and args.analysis_only:
        raise SystemExit("--data-only and --analysis-only are mutually exclusive")

    if not args.analysis_only:
        download_archive(args.archive_url, ARCHIVE_PATH)
        extract_archive(ARCHIVE_PATH)
    elif not _extracted_data_complete():
        raise SystemExit("Real data are missing; run without --analysis-only first")

    if args.data_only:
        return
    python = sys.executable if args.skip_install else prepare_environment(ROOT / ".venv")
    notebook = run_checks_and_analysis(Path(python), skip_tests=args.skip_tests)
    print(f"Reproduction complete. Open: {notebook}")


if __name__ == "__main__":
    main()
