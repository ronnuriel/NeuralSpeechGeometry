"""Geometry summaries and whole-trial permutation inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def trial_vectors(scores: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    """Average each trial over selected time bins without mixing trials."""
    scores = np.asarray(scores, dtype=float)
    time_mask = np.asarray(time_mask, dtype=bool)
    if scores.ndim != 3:
        raise ValueError("scores must have shape [trial, time, component]")
    if time_mask.shape != (scores.shape[1],) or not time_mask.any():
        raise ValueError("time_mask must select at least one time bin")
    return scores[:, time_mask, :].mean(axis=1)


def centroid_distance(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Euclidean distance between two condition centroids."""
    vectors = np.asarray(vectors, dtype=float)
    labels = np.asarray(labels)
    conditions = np.unique(labels)
    if conditions.size != 2:
        raise ValueError(f"Expected exactly two conditions, got {conditions.tolist()}")
    centroids = [vectors[labels == condition].mean(axis=0) for condition in conditions]
    return float(np.linalg.norm(centroids[0] - centroids[1]))


def time_resolved_centroid_distance(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Condition-centroid distance at every time bin."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    conditions = np.unique(labels)
    if conditions.size != 2:
        raise ValueError("Exactly two conditions are required")
    first = scores[labels == conditions[0]].mean(axis=0)
    second = scores[labels == conditions[1]].mean(axis=0)
    return np.linalg.norm(first - second, axis=1)


def word_geometry_correlation(
    vectors: np.ndarray,
    trials: pd.DataFrame,
    *,
    condition_column: str = "condition",
    word_column: str = "word",
) -> tuple[float, float, list[str]]:
    """Spearman correlation between within-condition word-distance patterns."""
    conditions = sorted(trials[condition_column].unique())
    if len(conditions) != 2:
        raise ValueError("Exactly two conditions are required")
    shared_words = sorted(
        set(trials.loc[trials[condition_column] == conditions[0], word_column]).intersection(
            trials.loc[trials[condition_column] == conditions[1], word_column]
        )
    )
    if len(shared_words) < 3:
        raise ValueError("At least three shared words are required")

    distance_vectors = []
    for condition in conditions:
        centroids = []
        for word in shared_words:
            mask = (
                (trials[condition_column].to_numpy() == condition)
                & (trials[word_column].to_numpy() == word)
            )
            centroids.append(vectors[mask].mean(axis=0))
        distance_vectors.append(pdist(np.stack(centroids), metric="euclidean"))
    statistic = spearmanr(distance_vectors[0], distance_vectors[1])
    return float(statistic.statistic), float(statistic.pvalue), shared_words


def word_centroid_tensor(
    vectors: np.ndarray,
    trials: pd.DataFrame,
    *,
    condition_column: str = "condition",
    word_column: str = "word",
) -> tuple[list[str], list[str], np.ndarray]:
    """Return matched word centroids as ``condition × word × feature``."""
    conditions = sorted(trials[condition_column].unique())
    if len(conditions) != 2:
        raise ValueError("Exactly two conditions are required")
    shared_words = sorted(
        set(trials.loc[trials[condition_column] == conditions[0], word_column]).intersection(
            trials.loc[trials[condition_column] == conditions[1], word_column]
        )
    )
    if len(shared_words) < 2:
        raise ValueError("At least two shared words are required")

    tensor = np.empty((2, len(shared_words), vectors.shape[1]), dtype=float)
    condition_values = trials[condition_column].to_numpy()
    word_values = trials[word_column].to_numpy()
    for condition_index, condition in enumerate(conditions):
        for word_index, word in enumerate(shared_words):
            mask = (condition_values == condition) & (word_values == word)
            tensor[condition_index, word_index] = vectors[mask].mean(axis=0)
    return conditions, shared_words, tensor


def word_distance_matrices(
    vectors: np.ndarray,
    trials: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return one Euclidean representational-distance matrix per condition."""
    conditions, words, centroids = word_centroid_tensor(vectors, trials)
    rdms = np.stack([squareform(pdist(condition_centroids)) for condition_centroids in centroids])
    return conditions, words, rdms


def word_geometry_scale_and_distortion(
    vectors: np.ndarray,
    trials: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    """Summarize word spread and condition-specific distortion after centering.

    Distortion is zero when the two conditions have identical word geometry up
    to a global translation. It remains sensitive to rotation and scale changes.
    """
    conditions, _, centroids = word_centroid_tensor(vectors, trials)
    spread = {
        condition: float(pdist(centroids[index]).mean())
        for index, condition in enumerate(conditions)
    }
    centered = centroids - centroids.mean(axis=1, keepdims=True)
    distortion = float(np.linalg.norm(centered[0] - centered[1], axis=1).mean())
    return spread, distortion


def condition_subspace_angles_deg(
    scores: np.ndarray,
    labels: np.ndarray,
    time_mask: np.ndarray,
    *,
    subspace_dimension: int = 3,
) -> np.ndarray:
    """Principal angles between condition-specific variance subspaces."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    conditions = np.unique(labels)
    if conditions.size != 2:
        raise ValueError("Exactly two conditions are required")
    if subspace_dimension >= scores.shape[2]:
        raise ValueError("subspace_dimension must be smaller than shared PCA dimension")

    bases = []
    for condition in conditions:
        matrix = scores[labels == condition][:, time_mask, :].reshape(-1, scores.shape[2])
        model = PCA(n_components=subspace_dimension, svd_solver="full").fit(matrix)
        bases.append(model.components_.T)
    return np.rad2deg(subspace_angles(bases[0], bases[1]))


@dataclass
class PermutationResult:
    observed: float
    null_distribution: np.ndarray
    p_value: float
    exchangeable_strata: int


def stratified_permutation_test(
    vectors: np.ndarray,
    trials: pd.DataFrame,
    *,
    condition_column: str = "condition",
    strata_columns: tuple[str, ...] | list[str] = ("session", "word"),
    n_permutations: int = 1000,
    random_state: int = 0,
) -> PermutationResult:
    """Test centroid distance by shuffling condition labels within trial strata.

    Whole trial vectors are permuted. Time bins are never treated as independent.
    Strata that contain only one condition are left unchanged and do not provide
    exchangeability for the condition contrast.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    for column in (condition_column, *strata_columns):
        if column not in trials:
            raise ValueError(f"Missing trial metadata column: {column}")

    labels = trials[condition_column].to_numpy(copy=True)
    observed = centroid_distance(vectors, labels)
    grouped = trials.groupby(list(strata_columns), sort=False, dropna=False).indices
    exchangeable = [
        np.asarray(indices)
        for indices in grouped.values()
        if np.unique(labels[np.asarray(indices)]).size > 1
    ]
    if not exchangeable:
        raise ValueError(
            "No stratum contains both conditions; a stratified label permutation is impossible"
        )

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled = labels.copy()
        for indices in exchangeable:
            shuffled[indices] = rng.permutation(shuffled[indices])
        null[permutation] = centroid_distance(vectors, shuffled)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1))
    return PermutationResult(
        observed=observed,
        null_distribution=null,
        p_value=p_value,
        exchangeable_strata=len(exchangeable),
    )
