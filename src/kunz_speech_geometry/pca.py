"""Shared-basis principal component analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .preprocess import balanced_trial_indices


@dataclass
class SharedPCAResult:
    """Scores and fitted parameters from one pooled PCA basis."""

    scores: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    fit_trial_indices: np.ndarray
    fit_time_mask: np.ndarray


@dataclass
class WordCentroidPCAResult:
    """Equal-weight condition-by-word centroids in one shared PCA basis."""

    metadata: pd.DataFrame
    centroids: np.ndarray
    scores: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray


def fit_shared_pca(
    rates: np.ndarray,
    labels: np.ndarray,
    *,
    n_components: int,
    fit_time_mask: np.ndarray | None = None,
    balance_conditions: bool = True,
    standardize_channels: bool = True,
    random_state: int = 0,
) -> SharedPCAResult:
    """Fit one scaler and PCA to pooled conditions, then transform every sample.

    Rows of the PCA matrix are trial-time observations and columns are channels.
    Conditions are only separated after projection.
    """
    rates = np.asarray(rates, dtype=float)
    labels = np.asarray(labels)
    if rates.ndim != 3:
        raise ValueError("rates must have shape [trial, time, channel]")
    if labels.shape != (rates.shape[0],):
        raise ValueError("labels must contain one value per trial")
    if fit_time_mask is None:
        fit_time_mask = np.ones(rates.shape[1], dtype=bool)
    fit_time_mask = np.asarray(fit_time_mask, dtype=bool)
    if fit_time_mask.shape != (rates.shape[1],) or not fit_time_mask.any():
        raise ValueError("fit_time_mask must select at least one time bin")

    if balance_conditions:
        fit_trial_indices = balanced_trial_indices(labels, random_state)
    else:
        fit_trial_indices = np.arange(rates.shape[0])

    # Channel scaling is fitted on all pooled observations and is therefore
    # independent of condition labels. PCA fitting may then use equal trial
    # counts so an over-represented condition does not dominate the basis.
    scaler_matrix = rates[:, fit_time_mask, :].reshape(-1, rates.shape[2])
    pca_matrix = rates[fit_trial_indices][:, fit_time_mask, :].reshape(-1, rates.shape[2])
    max_components = min(pca_matrix.shape)
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")

    scaler = StandardScaler(with_mean=True, with_std=standardize_channels)
    scaler.fit(scaler_matrix)
    fit_scaled = scaler.transform(pca_matrix)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    pca.fit(fit_scaled)

    all_matrix = rates.reshape(-1, rates.shape[2])
    all_scores = pca.transform(scaler.transform(all_matrix))
    scores = all_scores.reshape(rates.shape[0], rates.shape[1], n_components)
    scale = scaler.scale_ if scaler.scale_ is not None else np.ones(rates.shape[2])

    return SharedPCAResult(
        scores=scores,
        components=pca.components_.copy(),
        explained_variance_ratio=pca.explained_variance_ratio_.copy(),
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=np.asarray(scale).copy(),
        fit_trial_indices=fit_trial_indices,
        fit_time_mask=fit_time_mask.copy(),
    )


def standardized_trial_vectors(
    rates: np.ndarray,
    time_mask: np.ndarray,
    shared_pca: SharedPCAResult,
) -> np.ndarray:
    """Time-average trials and apply the pooled channel scaler, retaining all channels."""
    rates = np.asarray(rates, dtype=float)
    time_mask = np.asarray(time_mask, dtype=bool)
    if rates.ndim != 3:
        raise ValueError("rates must have shape [trial, time, channel]")
    if time_mask.shape != (rates.shape[1],) or not time_mask.any():
        raise ValueError("time_mask must select at least one time bin")
    if rates.shape[2] != shared_pca.scaler_mean.size:
        raise ValueError("Channel count does not match the fitted scaler")
    vectors = rates[:, time_mask, :].mean(axis=1)
    return (vectors - shared_pca.scaler_mean) / shared_pca.scaler_scale


def fit_word_centroid_pca(
    vectors: np.ndarray,
    trials: pd.DataFrame,
    *,
    n_components: int = 3,
    condition_column: str = "condition",
    word_column: str = "word",
) -> WordCentroidPCAResult:
    """Fit one PCA to equally weighted condition-by-word response centroids."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(trials):
        raise ValueError("vectors must have one row per trial")
    cells: list[dict[str, str]] = []
    centroids: list[np.ndarray] = []
    for condition in sorted(trials[condition_column].unique()):
        condition_words = sorted(
            trials.loc[trials[condition_column] == condition, word_column].unique()
        )
        for word in condition_words:
            mask = (
                (trials[condition_column].to_numpy() == condition)
                & (trials[word_column].to_numpy() == word)
            )
            cells.append({"condition": condition, "word": word})
            centroids.append(vectors[mask].mean(axis=0))
    centroid_matrix = np.stack(centroids)
    max_components = min(centroid_matrix.shape[0] - 1, centroid_matrix.shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")
    model = PCA(n_components=n_components, svd_solver="full").fit(centroid_matrix)
    return WordCentroidPCAResult(
        metadata=pd.DataFrame(cells),
        centroids=centroid_matrix,
        scores=model.transform(centroid_matrix),
        components=model.components_.copy(),
        explained_variance_ratio=model.explained_variance_ratio_.copy(),
    )
