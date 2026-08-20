"""Canonical in-memory data contract used by all analyses."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

REQUIRED_TRIAL_COLUMNS = {
    "participant",
    "session",
    "block",
    "trial_id",
    "condition",
    "condition_raw",
    "word",
}
REQUIRED_CHANNEL_COLUMNS = {"channel_id"}


@dataclass
class NeuralDataset:
    """Trial-aligned neural rates and matching metadata.

    The required axis order is ``[trial, time, channel]``. Adapters for source
    files must make axis transpositions explicit before constructing this class.
    """

    rates: np.ndarray
    time_ms: np.ndarray
    trials: pd.DataFrame
    channels: pd.DataFrame
    feature_name: str
    units: str
    alignment_event: str

    def validate(self) -> NeuralDataset:
        rates = np.asarray(self.rates)
        time_ms = np.asarray(self.time_ms)

        if rates.ndim != 3:
            raise ValueError(
                f"rates must have shape [trial, time, channel], got {rates.shape}"
            )
        if time_ms.ndim != 1:
            raise ValueError(f"time_ms must be one-dimensional, got {time_ms.shape}")
        if rates.shape[1] != time_ms.size:
            raise ValueError(
                f"rates has {rates.shape[1]} time bins but time_ms has {time_ms.size}"
            )
        if rates.shape[0] != len(self.trials):
            raise ValueError(
                f"rates has {rates.shape[0]} trials but metadata has {len(self.trials)} rows"
            )
        if rates.shape[2] != len(self.channels):
            raise ValueError(
                f"rates has {rates.shape[2]} channels but metadata has {len(self.channels)} rows"
            )
        missing_trials = REQUIRED_TRIAL_COLUMNS.difference(self.trials.columns)
        if missing_trials:
            raise ValueError(f"trials is missing columns: {sorted(missing_trials)}")
        missing_channels = REQUIRED_CHANNEL_COLUMNS.difference(self.channels.columns)
        if missing_channels:
            raise ValueError(f"channels is missing columns: {sorted(missing_channels)}")
        if not np.all(np.isfinite(rates)):
            raise ValueError("rates contains NaN or infinite values")
        if not np.all(np.isfinite(time_ms)):
            raise ValueError("time_ms contains NaN or infinite values")
        if time_ms.size > 1 and not np.all(np.diff(time_ms) > 0):
            raise ValueError("time_ms must be strictly increasing")
        if self.trials["trial_id"].duplicated().any():
            raise ValueError("trial_id values must be unique")
        if self.channels["channel_id"].duplicated().any():
            raise ValueError("channel_id values must be unique")
        if not self.feature_name:
            raise ValueError("feature_name must be documented")
        if not self.units:
            raise ValueError("units must be documented")
        if not self.alignment_event:
            raise ValueError("alignment_event must be documented")
        return self

    def subset_conditions(self, conditions: Iterable[str]) -> NeuralDataset:
        """Return a copy containing only the requested canonical conditions."""
        requested = list(conditions)
        mask = self.trials["condition"].isin(requested).to_numpy()
        missing = set(requested).difference(self.trials.loc[mask, "condition"].unique())
        if missing:
            raise ValueError(f"Conditions not present: {sorted(missing)}")
        subset = replace(
            self,
            rates=self.rates[mask].copy(),
            trials=self.trials.loc[mask].reset_index(drop=True).copy(),
            channels=self.channels.copy(),
            time_ms=self.time_ms.copy(),
        )
        return subset.validate()

    def with_rates(self, rates: np.ndarray) -> NeuralDataset:
        """Return a validated copy with transformed neural values."""
        return replace(self, rates=np.asarray(rates)).validate()

    @property
    def bin_width_ms(self) -> float:
        if self.time_ms.size < 2:
            raise ValueError("At least two time bins are required")
        return float(np.median(np.diff(self.time_ms)))
