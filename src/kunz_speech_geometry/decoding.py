"""Leakage-safe trial decoding with block-aware PCA model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


@dataclass
class FixedPCDecodingResult:
    """Block-held-out scores for predeclared PCA dimensionalities."""

    summary: pd.DataFrame
    fold_scores: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class NestedPCDecodingResult:
    """Outer block-held-out predictions after inner-CV PC selection."""

    fold_scores: pd.DataFrame
    inner_scores: pd.DataFrame
    predictions: pd.DataFrame
    pooled_balanced_accuracy: float
    mean_block_balanced_accuracy: float


def flatten_trial_epochs(rates: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    """Return one row-major time-by-channel vector per selected trial epoch."""
    rates = np.asarray(rates, dtype=float)
    time_mask = np.asarray(time_mask, dtype=bool)
    if rates.ndim != 3:
        raise ValueError("rates must have shape [trial, time, channel]")
    if time_mask.shape != (rates.shape[1],) or not time_mask.any():
        raise ValueError("time_mask must select at least one time bin")
    if not np.isfinite(rates).all():
        raise ValueError("rates must contain only finite values")
    return rates[:, time_mask, :].reshape(rates.shape[0], -1)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _validate_decoding_inputs(
    vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pc_candidates: tuple[int, ...] | list[int],
    *,
    minimum_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    vectors = np.asarray(vectors, dtype=float)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    candidates = tuple(sorted(set(int(value) for value in pc_candidates)))
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [trial, feature]")
    if labels.shape != (vectors.shape[0],) or groups.shape != (vectors.shape[0],):
        raise ValueError("labels and groups must contain one value per trial")
    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")
    if np.unique(labels).size != 2:
        raise ValueError("Exactly two condition labels are required")
    if np.unique(groups).size < minimum_groups:
        raise ValueError(f"At least {minimum_groups} groups are required")
    if not candidates or candidates[0] < 1:
        raise ValueError("pc_candidates must contain positive integers")
    if candidates[-1] > min(vectors.shape):
        raise ValueError("A PC candidate exceeds the available matrix rank bound")
    return vectors, labels, groups, candidates


def _fit_fold_projection(
    vectors: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    max_components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler and PCA on training rows only, then project held-out rows."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(vectors[train_indices])
    test_scaled = scaler.transform(vectors[test_indices])
    max_allowed = min(train_scaled.shape)
    if max_components > max_allowed:
        raise ValueError(
            f"Requested {max_components} PCs, but this training fold allows {max_allowed}"
        )
    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    return train_scores, test_scores


def fixed_pc_block_cv(
    vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    pc_candidates: tuple[int, ...] | list[int] = (1, 2, 3, 5, 10, 20),
    random_state: int = 0,
) -> FixedPCDecodingResult:
    """Compare fixed PC counts with a fresh scaler/PCA in every held-out block.

    The largest requested PCA is fitted only on the training rows of each fold.
    Smaller candidate models use prefixes of that same training-only basis. A
    separate logistic classifier is then fitted for every candidate.
    """
    vectors, labels, groups, candidates = _validate_decoding_inputs(
        vectors,
        labels,
        groups,
        pc_candidates,
        minimum_groups=2,
    )
    splitter = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(vectors, labels, groups)
    ):
        train_scores, test_scores = _fit_fold_projection(
            vectors,
            train_indices,
            test_indices,
            max_components=candidates[-1],
            random_state=random_state + fold_index,
        )
        test_group_values = np.unique(groups[test_indices])
        if test_group_values.size != 1:
            raise AssertionError("LeaveOneGroupOut produced more than one test group")
        test_group = _python_scalar(test_group_values[0])

        for n_components in candidates:
            classifier = LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=random_state + fold_index,
            )
            classifier.fit(train_scores[:, :n_components], labels[train_indices])
            predicted = classifier.predict(test_scores[:, :n_components])
            score = balanced_accuracy_score(labels[test_indices], predicted)
            fold_rows.append(
                {
                    "n_components": n_components,
                    "test_group": test_group,
                    "n_train": train_indices.size,
                    "n_test": test_indices.size,
                    "balanced_accuracy": float(score),
                }
            )
            prediction_rows.extend(
                {
                    "trial_index": int(trial_index),
                    "n_components": n_components,
                    "test_group": test_group,
                    "true_label": _python_scalar(labels[trial_index]),
                    "predicted_label": _python_scalar(prediction),
                }
                for trial_index, prediction in zip(test_indices, predicted, strict=True)
            )

    fold_scores = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary_rows = []
    for n_components in candidates:
        candidate_folds = fold_scores.loc[fold_scores["n_components"] == n_components]
        candidate_predictions = predictions.loc[
            predictions["n_components"] == n_components
        ]
        summary_rows.append(
            {
                "n_components": n_components,
                "mean_block_balanced_accuracy": candidate_folds[
                    "balanced_accuracy"
                ].mean(),
                "std_block_balanced_accuracy": candidate_folds[
                    "balanced_accuracy"
                ].std(ddof=1),
                "pooled_balanced_accuracy": balanced_accuracy_score(
                    candidate_predictions["true_label"],
                    candidate_predictions["predicted_label"],
                ),
            }
        )
    return FixedPCDecodingResult(
        summary=pd.DataFrame(summary_rows),
        fold_scores=fold_scores,
        predictions=predictions,
    )


def nested_pc_block_cv(
    vectors: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    pc_candidates: tuple[int, ...] | list[int] = (1, 2, 3, 5, 10, 20),
    random_state: int = 0,
) -> NestedPCDecodingResult:
    """Select the PC count in inner block CV and evaluate on an unseen outer block."""
    vectors, labels, groups, candidates = _validate_decoding_inputs(
        vectors,
        labels,
        groups,
        pc_candidates,
        minimum_groups=3,
    )
    splitter = LeaveOneGroupOut()
    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(vectors, labels, groups)
    ):
        test_group = _python_scalar(np.unique(groups[test_indices])[0])
        inner = fixed_pc_block_cv(
            vectors[train_indices],
            labels[train_indices],
            groups[train_indices],
            pc_candidates=candidates,
            random_state=random_state + 100 * (fold_index + 1),
        )
        ranked = inner.summary.sort_values(
            ["mean_block_balanced_accuracy", "n_components"],
            ascending=[False, True],
        )
        selected = int(ranked.iloc[0]["n_components"])
        selected_inner_score = float(ranked.iloc[0]["mean_block_balanced_accuracy"])
        for row in inner.summary.to_dict("records"):
            inner_rows.append({"outer_test_group": test_group, **row})

        train_scores, test_scores = _fit_fold_projection(
            vectors,
            train_indices,
            test_indices,
            max_components=candidates[-1],
            random_state=random_state + fold_index,
        )
        classifier = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=random_state + fold_index,
        )
        classifier.fit(train_scores[:, :selected], labels[train_indices])
        predicted = classifier.predict(test_scores[:, :selected])
        outer_score = balanced_accuracy_score(labels[test_indices], predicted)
        fold_rows.append(
            {
                "test_group": test_group,
                "n_train": train_indices.size,
                "n_test": test_indices.size,
                "selected_n_components": selected,
                "best_inner_mean_balanced_accuracy": selected_inner_score,
                "outer_balanced_accuracy": float(outer_score),
            }
        )
        prediction_rows.extend(
            {
                "trial_index": int(trial_index),
                "test_group": test_group,
                "selected_n_components": selected,
                "true_label": _python_scalar(labels[trial_index]),
                "predicted_label": _python_scalar(prediction),
            }
            for trial_index, prediction in zip(test_indices, predicted, strict=True)
        )

    fold_scores = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows).sort_values("trial_index").reset_index(drop=True)
    return NestedPCDecodingResult(
        fold_scores=fold_scores,
        inner_scores=pd.DataFrame(inner_rows),
        predictions=predictions,
        pooled_balanced_accuracy=float(
            balanced_accuracy_score(
                predictions["true_label"], predictions["predicted_label"]
            )
        ),
        mean_block_balanced_accuracy=float(fold_scores["outer_balanced_accuracy"].mean()),
    )
