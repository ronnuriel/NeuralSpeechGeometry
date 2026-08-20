"""Conservative MATLAB file inspection utilities.

The real Kunz adapter will be added only after inspecting an actual source file.
This module deliberately avoids guessing nested field names or axis order.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import pandas as pd
from scipy.io import whosmat


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a source-file checksum without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hdf5_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                rows.append(
                    {
                        "name": name,
                        "shape_on_disk": tuple(item.shape),
                        "class_or_dtype": str(item.dtype),
                        "storage": "MATLAB v7.3 / HDF5",
                    }
                )

        handle.visititems(visitor)
    return rows


def inspect_mat_file(path: str | Path, *, include_checksum: bool = False) -> pd.DataFrame:
    """List variable paths, shapes, and storage type without loading full arrays."""
    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    if h5py.is_hdf5(mat_path):
        rows = _hdf5_rows(mat_path)
    else:
        rows = [
            {
                "name": name,
                "shape_on_disk": tuple(shape),
                "class_or_dtype": matlab_class,
                "storage": "MATLAB <= v7.2",
            }
            for name, shape, matlab_class in whosmat(mat_path)
        ]
    frame = pd.DataFrame(rows)
    frame.insert(0, "file", mat_path.name)
    if include_checksum:
        frame["sha256"] = sha256_file(mat_path)
    return frame


def audit_directory(raw_dir: str | Path, *, include_checksum: bool = False) -> pd.DataFrame:
    """Audit every `.mat` file recursively under a raw-data directory."""
    paths = sorted(Path(raw_dir).rglob("*.mat"))
    if not paths:
        return pd.DataFrame(
            columns=["file", "name", "shape_on_disk", "class_or_dtype", "storage"]
        )
    return pd.concat(
        [inspect_mat_file(path, include_checksum=include_checksum) for path in paths],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MATLAB variables without loading full data"
    )
    parser.add_argument("path", type=Path, help="A .mat file or directory")
    parser.add_argument("--checksum", action="store_true", help="Compute SHA-256 (can be slow)")
    args = parser.parse_args()
    if args.path.is_dir():
        result = audit_directory(args.path, include_checksum=args.checksum)
    else:
        result = inspect_mat_file(args.path, include_checksum=args.checksum)
    if result.empty:
        print("No .mat files found.")
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
