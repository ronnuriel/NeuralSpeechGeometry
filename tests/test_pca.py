import numpy as np

from kunz_speech_geometry.pca import (
    fit_shared_pca,
    fit_word_centroid_pca,
    standardized_trial_vectors,
)
from kunz_speech_geometry.preprocess import smooth_and_baseline, window_mask
from kunz_speech_geometry.synthetic import make_synthetic_dataset


def test_shared_pca_uses_one_basis_and_balances_fit_trials():
    dataset = make_synthetic_dataset(
        n_channels=20,
        n_trials_per_word_condition=3,
        random_state=11,
    )
    processed = smooth_and_baseline(
        dataset,
        smoothing_sigma_ms=40,
        baseline_window_ms=(-500, -100),
    )
    fit_mask = window_mask(processed.time_ms, (0, 700))
    labels = processed.trials["condition"].to_numpy()
    result = fit_shared_pca(
        processed.rates,
        labels,
        n_components=6,
        fit_time_mask=fit_mask,
        balance_conditions=True,
        random_state=3,
    )
    assert result.scores.shape == (*processed.rates.shape[:2], 6)
    assert result.components.shape == (6, processed.rates.shape[2])
    fit_labels, counts = np.unique(labels[result.fit_trial_indices], return_counts=True)
    assert fit_labels.size == 2
    assert counts[0] == counts[1]
    assert np.isclose(np.linalg.norm(result.components[0]), 1.0)


def test_word_centroid_pca_has_one_row_per_condition_word_cell():
    dataset = make_synthetic_dataset(
        n_channels=18,
        n_trials_per_word_condition=2,
        random_state=5,
    )
    mask = window_mask(dataset.time_ms, (0, 500))
    labels = dataset.trials["condition"].to_numpy()
    shared = fit_shared_pca(
        dataset.rates,
        labels,
        n_components=5,
        fit_time_mask=mask,
        random_state=5,
    )
    vectors = standardized_trial_vectors(dataset.rates, mask, shared)
    word_pca = fit_word_centroid_pca(vectors, dataset.trials, n_components=3)
    assert word_pca.scores.shape == (14, 3)
    assert set(word_pca.metadata.columns) == {"condition", "word"}
