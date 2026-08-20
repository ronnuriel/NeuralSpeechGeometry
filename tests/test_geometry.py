import numpy as np

from kunz_speech_geometry.geometry import (
    centroid_distance,
    stratified_permutation_test,
    word_distance_matrices,
    word_geometry_correlation,
    word_geometry_scale_and_distortion,
)
from kunz_speech_geometry.pca import fit_shared_pca, standardized_trial_vectors
from kunz_speech_geometry.preprocess import smooth_and_baseline, window_mask
from kunz_speech_geometry.synthetic import make_synthetic_dataset


def _analysis_vectors():
    dataset = make_synthetic_dataset(
        n_channels=24,
        n_trials_per_word_condition=4,
        random_state=23,
    )
    processed = smooth_and_baseline(
        dataset,
        smoothing_sigma_ms=60,
        baseline_window_ms=(-500, -100),
    )
    mask = window_mask(processed.time_ms, (0, 800))
    labels = processed.trials["condition"].to_numpy()
    pca = fit_shared_pca(
        processed.rates,
        labels,
        n_components=8,
        fit_time_mask=mask,
        random_state=23,
    )
    vectors = standardized_trial_vectors(processed.rates, mask, pca)
    return dataset, vectors


def test_geometry_metrics_are_finite():
    dataset, vectors = _analysis_vectors()
    labels = dataset.trials["condition"].to_numpy()
    assert centroid_distance(vectors, labels) > 0
    correlation, p_value, words = word_geometry_correlation(
        vectors,
        dataset.trials,
    )
    assert np.isfinite(correlation)
    assert np.isfinite(p_value)
    assert len(words) == 7
    conditions, rdm_words, rdms = word_distance_matrices(vectors, dataset.trials)
    assert len(conditions) == 2
    assert rdm_words == words
    assert rdms.shape == (2, 7, 7)
    assert np.allclose(np.diagonal(rdms, axis1=1, axis2=2), 0)
    spread, distortion = word_geometry_scale_and_distortion(vectors, dataset.trials)
    assert set(spread) == set(conditions)
    assert all(value > 0 for value in spread.values())
    assert distortion >= 0


def test_permutation_is_trial_level_and_deterministic():
    dataset, vectors = _analysis_vectors()
    first = stratified_permutation_test(
        vectors,
        dataset.trials,
        strata_columns=("session", "block", "word"),
        n_permutations=50,
        random_state=9,
    )
    second = stratified_permutation_test(
        vectors,
        dataset.trials,
        strata_columns=("session", "block", "word"),
        n_permutations=50,
        random_state=9,
    )
    assert first.exchangeable_strata == 14
    assert np.array_equal(first.null_distribution, second.null_distribution)
    assert 0 < first.p_value <= 1
