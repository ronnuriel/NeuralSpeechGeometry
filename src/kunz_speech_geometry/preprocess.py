"""Preprocessing that preserves trials as the independent sampling unit."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .schema import NeuralDataset


def window_mask(time_ms: np.ndarray, window_ms: tuple[float, float] | list[float]) -> np.ndarray:
    """Return an inclusive Boolean mask for a time window."""
    start, stop = map(float, window_ms)
    if start >= stop:
        raise ValueError(f"Window start must be before stop, got {window_ms}")
    mask = (np.asarray(time_ms) >= start) & (np.asarray(time_ms) <= stop)
    if not mask.any():
        raise ValueError(f"No time bins fall inside window {window_ms}")
    return mask


def smooth_and_baseline(
    dataset: NeuralDataset,
    *,
    smoothing_sigma_ms: float,
    baseline_window_ms: tuple[float, float] | list[float],
    baseline_correct: bool = True,
) -> NeuralDataset:
    """Smooth across time and optionally subtract each trial/channel baseline."""
    dataset.validate()
    rates = dataset.rates.astype(float, copy=True)
    if smoothing_sigma_ms < 0:
        raise ValueError("smoothing_sigma_ms must be non-negative")
    if smoothing_sigma_ms > 0:
        sigma_bins = smoothing_sigma_ms / dataset.bin_width_ms
        rates = gaussian_filter1d(rates, sigma=sigma_bins, axis=1, mode="nearest")
    if baseline_correct:
        mask = window_mask(dataset.time_ms, baseline_window_ms)
        baseline = rates[:, mask, :].mean(axis=1, keepdims=True)
        rates = rates - baseline
    return dataset.with_rates(rates)


def balanced_trial_indices(labels: np.ndarray, random_state: int) -> np.ndarray:
    """Sample equal trial counts from each condition for fitting a shared basis."""
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        raise ValueError("At least two conditions are required")
    target = int(counts.min())
    rng = np.random.default_rng(random_state)
    chosen = [
        rng.choice(np.flatnonzero(labels == label), target, replace=False) for label in unique
    ]
    return np.sort(np.concatenate(chosen))
