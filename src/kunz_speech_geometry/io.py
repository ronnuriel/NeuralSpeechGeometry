"""MATLAB audit tools and an explicit adapter for the Kunz interleaved task."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from .schema import NeuralDataset

WORDS = ("ban", "choice", "day", "feel", "kite", "though", "were")
INTERLEAVED_METADATA_FIELDS = {
    "blockNum",
    "goTrialEpochs",
    "delayTrialEpochs",
    "trialCues",
    "cueList",
    "chanSets",
    "chanSetNames",
    "binSize",
}


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


def _matlab_strings(values: Any) -> list[str]:
    """Flatten either row- or column-oriented MATLAB cell strings."""
    return [str(value).strip() for value in np.atleast_1d(values).reshape(-1)]


def _participant_and_session(path: Path) -> tuple[str, str]:
    session = path.name.split("_", maxsplit=1)[0]
    participant = session.split(".", maxsplit=1)[0].upper()
    if participant not in {"T12", "T15", "T16", "T17"}:
        raise ValueError(f"Cannot infer a supported participant from {path.name}")
    return participant, session


def _parse_interleaved_cue(raw_cue: str, participant: str) -> tuple[str, str] | None:
    """Map the source cue text to a canonical condition and one of seven words."""
    normalized = raw_cue.lower().replace("\\n", " ").replace("_", "")
    if "nothing" in normalized:
        return None
    word = next((candidate for candidate in WORDS if normalized.startswith(candidate)), None)
    if word is None:
        raise ValueError(f"Unrecognized interleaved cue: {raw_cue!r}")

    if "mimed" in normalized:
        condition = "attempted_mimed"
    elif "attempted" in normalized:
        condition = "attempted_speech" if participant == "T17" else "attempted_vocalized"
    elif "passivelistening" in normalized or ",listen" in normalized:
        condition = "passive_listening"
    elif "imaginedlistening" in normalized:
        condition = "imagined_listening"
    elif "imagined" in normalized:
        condition = "imagined_motoric" if participant == "T12" else "imagined_auditory"
    elif participant == "T12" and "$0" in normalized:
        # T12 uses an untagged spoken-word cue for passive listening.
        condition = "passive_listening"
    else:
        raise ValueError(f"Unrecognized condition in cue: {raw_cue!r}")
    return condition, word


def _channel_selection(
    channel_sets: Any,
    channel_set_names: Any,
    requested: str | list[str] | tuple[str, ...] | None,
    n_channels: int,
) -> tuple[np.ndarray, dict[int, str]]:
    names = _matlab_strings(channel_set_names)
    sets = np.atleast_1d(channel_sets).reshape(-1)
    if len(names) != len(sets):
        raise ValueError("chanSets and chanSetNames have different lengths")

    source_sets: dict[str, np.ndarray] = {}
    channel_to_set: dict[int, str] = {}
    for name, values in zip(names, sets, strict=True):
        indices = np.atleast_1d(values).astype(int).reshape(-1) - 1
        if indices.size == 0 or indices.min() < 0 or indices.max() >= n_channels:
            raise ValueError(f"Invalid MATLAB channel indices for {name!r}")
        source_sets[name.casefold()] = indices
        for index in indices:
            channel_to_set[int(index)] = name

    if requested is None:
        selected = np.arange(n_channels)
    else:
        requested_names = [requested] if isinstance(requested, str) else list(requested)
        missing = [name for name in requested_names if name.casefold() not in source_sets]
        if missing:
            raise ValueError(f"Unknown channel sets {missing}; available: {names}")
        selected = np.unique(
            np.concatenate([source_sets[name.casefold()] for name in requested_names])
        )
    return selected, channel_to_set


def load_interleaved_mat(
    path: str | Path,
    *,
    feature: str = "binnedTX",
    channel_sets: str | list[str] | tuple[str, ...] | None = None,
    epoch_ms: tuple[int, int] = (-500, 1000),
    block_center: bool = True,
) -> NeuralDataset:
    """Load one public interleaved-task file into the canonical trial tensor.

    MATLAB cue, epoch, and channel indices are converted explicitly from 1-based
    to 0-based indexing. Epochs are half-open in Python. For T12 only, passive
    listening is aligned to the delay start, matching the authors' Figure 5 code;
    all other included cues use the go-period start.
    """
    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)
    if feature not in {"binnedTX", "spikePow"}:
        raise ValueError("feature must be 'binnedTX' or 'spikePow'")

    requested_fields = sorted(INTERLEAVED_METADATA_FIELDS | {feature})
    source = loadmat(
        mat_path,
        squeeze_me=True,
        variable_names=requested_fields,
    )
    missing = (INTERLEAVED_METADATA_FIELDS | {feature}).difference(source)
    if missing:
        raise ValueError(f"Missing interleaved-task fields: {sorted(missing)}")

    participant, session = _participant_and_session(mat_path)
    bin_size_ms = int(np.asarray(source["binSize"]).item())
    start_ms, stop_ms = map(int, epoch_ms)
    if start_ms >= stop_ms or start_ms % bin_size_ms or stop_ms % bin_size_ms:
        raise ValueError(f"epoch_ms must be ordered multiples of {bin_size_ms} ms")
    relative_bins = np.arange(start_ms // bin_size_ms, stop_ms // bin_size_ms)
    time_ms = relative_bins.astype(float) * bin_size_ms

    feature_matrix = np.asarray(source[feature])
    if feature_matrix.ndim != 2:
        raise ValueError(f"{feature} must be [continuous time, channel]")
    selected, channel_to_set = _channel_selection(
        source["chanSets"], source["chanSetNames"], channel_sets, feature_matrix.shape[1]
    )
    continuous = feature_matrix[:, selected].astype(float, copy=True)
    block_numbers = np.atleast_1d(source["blockNum"]).astype(int).reshape(-1)
    if continuous.shape[0] != block_numbers.size:
        raise ValueError("Neural feature rows and blockNum length differ")
    if block_center:
        for block in np.unique(block_numbers):
            mask = block_numbers == block
            continuous[mask] -= continuous[mask].mean(axis=0, keepdims=True)

    cues = _matlab_strings(source["cueList"])
    trial_cues = np.atleast_1d(source["trialCues"]).astype(int).reshape(-1) - 1
    go_epochs = np.asarray(source["goTrialEpochs"]).reshape(-1, 2).astype(int) - 1
    delay_epochs = np.asarray(source["delayTrialEpochs"]).reshape(-1, 2).astype(int) - 1
    if not (trial_cues.size == go_epochs.shape[0] == delay_epochs.shape[0]):
        raise ValueError("Trial cue and epoch counts differ")

    trial_arrays: list[np.ndarray] = []
    trial_rows: list[dict[str, object]] = []
    for trial_index, cue_index in enumerate(trial_cues):
        if cue_index < 0 or cue_index >= len(cues):
            raise ValueError(f"Trial {trial_index} has invalid cue index {cue_index + 1}")
        parsed = _parse_interleaved_cue(cues[cue_index], participant)
        if parsed is None:
            continue
        condition, word = parsed
        uses_t12_audio_onset = participant == "T12" and condition == "passive_listening"
        alignment = "delay_start" if uses_t12_audio_onset else "go_start"
        anchor = delay_epochs[trial_index, 0] if uses_t12_audio_onset else go_epochs[trial_index, 0]
        indices = anchor + relative_bins
        if indices.min() < 0 or indices.max() >= continuous.shape[0]:
            raise ValueError(f"Trial {trial_index} epoch falls outside the recording")
        block = int(block_numbers[anchor])
        if not np.all(block_numbers[indices] == block):
            raise ValueError(f"Trial {trial_index} epoch crosses a recording block boundary")
        trial_arrays.append(continuous[indices])
        trial_rows.append(
            {
                "participant": participant,
                "session": session,
                "block": block,
                "trial_id": f"{session}_trial_{trial_index:04d}",
                "source_trial_index": trial_index,
                "condition": condition,
                "condition_raw": cues[cue_index],
                "word": word,
                "alignment_event_raw": alignment,
            }
        )

    if not trial_arrays:
        raise ValueError("No speech trials were found")
    channels = pd.DataFrame(
        {
            "channel_id": [f"ch{index + 1:03d}" for index in selected],
            "source_channel_index_matlab": selected + 1,
            "channel_set": [channel_to_set.get(int(index), "unknown") for index in selected],
        }
    )
    units = "counts/bin" if feature == "binnedTX" else "microvolts^2"
    feature_name = f"{feature}_block_centered" if block_center else feature
    dataset = NeuralDataset(
        rates=np.stack(trial_arrays),
        time_ms=time_ms,
        trials=pd.DataFrame(trial_rows),
        channels=channels,
        feature_name=feature_name,
        units=units,
        alignment_event="condition-specific behavior onset",
    )
    return dataset.validate()


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
