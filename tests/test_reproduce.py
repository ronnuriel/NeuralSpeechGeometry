import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("reproduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def _tiny_archive(path: Path, member: str = "dataset/example.mat") -> bytes:
    payload = b"small neural fixture"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload)
    return payload


def test_download_verify_and_extract_from_file_url(tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    payload = _tiny_archive(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "raw" / "downloads" / "data.zip"
    extract_root = tmp_path / "raw"
    monkeypatch.setattr(reproduce, "ARCHIVE_SIZE", source.stat().st_size)
    monkeypatch.setattr(reproduce, "ARCHIVE_SHA256", digest)
    monkeypatch.setattr(reproduce, "EXTRACT_ROOT", extract_root)
    monkeypatch.setattr(
        reproduce, "EXPECTED_EXTRACTED_FILES", {"dataset/example.mat": len(payload)}
    )

    reproduce.download_archive(source.as_uri(), destination)
    reproduce.extract_archive(destination)

    assert destination.read_bytes() == source.read_bytes()
    assert (extract_root / "dataset" / "example.mat").read_bytes() == payload


def test_safe_extraction_rejects_parent_path(tmp_path, monkeypatch):
    source = tmp_path / "unsafe.zip"
    _tiny_archive(source, member="../escape.mat")
    monkeypatch.setattr(reproduce, "EXTRACT_ROOT", tmp_path / "raw")
    monkeypatch.setattr(reproduce, "EXPECTED_EXTRACTED_FILES", {"expected.mat": 1})

    with pytest.raises(RuntimeError, match="Unsafe path"):
        reproduce.extract_archive(source)
