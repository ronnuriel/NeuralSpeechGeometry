"""Deterministic synthetic dataset for executable examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import NeuralDataset

DEFAULT_WORDS = ("ban", "choice", "day", "feel", "kite", "though", "were")
DEFAULT_CONDITIONS = ("attempted_vocalized", "passive_listening")


def _gaussian(time_ms: np.ndarray, center_ms: float, width_ms: float) -> np.ndarray:
    return np.exp(-0.5 * ((time_ms - center_ms) / width_ms) ** 2)


def make_synthetic_dataset(
    *,
    participant: str = "T15",
    n_channels: int = 48,
    n_trials_per_word_condition: int = 8,
    bin_width_ms: int = 20,
    random_state: int = 20250821,
) -> NeuralDataset:
    """Create a small, structured stand-in for the unavailable real recordings.

    The generator includes shared word geometry, a stronger attempted response,
    a modest condition-specific axis, session drift, and trial noise. It is not
    intended to model the participants or reproduce the paper numerically.
    """
    if n_channels < 8:
        raise ValueError("n_channels must be at least 8")
    if n_trials_per_word_condition < 2:
        raise ValueError("At least two trials per word and condition are required")

    rng = np.random.default_rng(random_state)
    time_ms = np.arange(-500, 1000 + bin_width_ms, bin_width_ms, dtype=float)
    latent_dim = 6

    channel_loadings, _ = np.linalg.qr(rng.normal(size=(n_channels, latent_dim)))
    word_vectors = rng.normal(size=(len(DEFAULT_WORDS), latent_dim))
    word_vectors /= np.linalg.norm(word_vectors, axis=1, keepdims=True)
    condition_axis = rng.normal(size=latent_dim)
    condition_axis /= np.linalg.norm(condition_axis)
    shared_axis = rng.normal(size=latent_dim)
    shared_axis /= np.linalg.norm(shared_axis)

    attempted_envelope = _gaussian(time_ms, center_ms=300, width_ms=210)
    listening_envelope = _gaussian(time_ms, center_ms=220, width_ms=170)
    condition_envelope = _gaussian(time_ms, center_ms=420, width_ms=230)
    onset_envelope = _gaussian(time_ms, center_ms=80, width_ms=100)

    rows: list[dict[str, object]] = []
    trial_arrays: list[np.ndarray] = []
    trial_id = 0

    for condition in DEFAULT_CONDITIONS:
        for word_index, word in enumerate(DEFAULT_WORDS):
            for repetition in range(n_trials_per_word_condition):
                session_index = repetition % 2
                session = f"synthetic_session_{session_index + 1}"
                # Both conditions share a block label, matching the recommended
                # interleaved design for a condition-label permutation.
                block = f"synthetic_interleaved_block_{session_index + 1}"

                amplitude = 7.0 if condition == "attempted_vocalized" else 4.6
                envelope = (
                    attempted_envelope
                    if condition == "attempted_vocalized"
                    else listening_envelope
                )
                signed_condition = 1.0 if condition == "attempted_vocalized" else -0.35

                latent = amplitude * envelope[:, None] * word_vectors[word_index]
                latent += 2.0 * onset_envelope[:, None] * shared_axis
                latent += (
                    1.4
                    * signed_condition
                    * condition_envelope[:, None]
                    * condition_axis
                )

                session_drift = rng.normal(scale=0.25, size=n_channels) * session_index
                trial_gain = rng.normal(loc=1.0, scale=0.08)
                signal = trial_gain * latent @ channel_loadings.T
                noise = rng.normal(scale=0.85, size=(time_ms.size, n_channels))
                rates = np.clip(8.0 + signal + session_drift + noise, a_min=0.0, a_max=None)

                trial_arrays.append(rates)
                rows.append(
                    {
                        "participant": participant,
                        "session": session,
                        "block": block,
                        "trial_id": f"synthetic_{trial_id:04d}",
                        "condition": condition,
                        "condition_raw": condition,
                        "word": word,
                        "quality_flag": "include",
                    }
                )
                trial_id += 1

    channels = pd.DataFrame(
        {
            "channel_id": [f"ch_{index:03d}" for index in range(n_channels)],
            "array": [f"array_{index // 64 + 1}" for index in range(n_channels)],
            "quality_flag": "include",
        }
    )
    dataset = NeuralDataset(
        rates=np.stack(trial_arrays),
        time_ms=time_ms,
        trials=pd.DataFrame(rows),
        channels=channels,
        feature_name="synthetic_threshold_crossing_rate",
        units="a.u.",
        alignment_event="synthetic_behavior_onset",
    )
    return dataset.validate()
