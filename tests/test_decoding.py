import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

import kunz_speech_geometry.decoding as decoding
from kunz_speech_geometry.decoding import (
    fixed_pc_block_cv,
    flatten_trial_epochs,
    nested_pc_block_cv,
)


def _decodable_grouped_data(random_state: int = 5):
    rng = np.random.default_rng(random_state)
    n_groups = 5
    trials_per_group = 20
    groups = np.repeat(np.arange(n_groups), trials_per_group)
    labels = np.tile(np.repeat([0, 1], trials_per_group // 2), n_groups)
    vectors = rng.normal(0, 1, size=(groups.size, 40))
    vectors[:, :12] += labels[:, None] * 2.0
    vectors += rng.normal(0, 0.15, size=(n_groups, 40))[groups]
    return vectors, labels, groups


def test_flatten_trial_epochs_uses_time_major_channel_order():
    rates = np.arange(3 * 4 * 2).reshape(3, 4, 2)
    mask = np.array([False, True, True, False])
    vectors = flatten_trial_epochs(rates, mask)
    np.testing.assert_array_equal(vectors, rates[:, mask, :].reshape(3, 4))


def test_fixed_pc_cv_fits_scaler_without_held_out_group(monkeypatch):
    vectors, labels, groups = _decodable_grouped_data()
    vectors[:, 0] = groups
    seen_group_values = []

    class RecordingScaler(SklearnStandardScaler):
        def fit(self, values, y=None, sample_weight=None):
            seen_group_values.append(set(np.unique(values[:, 0]).astype(int)))
            return super().fit(values, y=y, sample_weight=sample_weight)

    monkeypatch.setattr(decoding, "StandardScaler", RecordingScaler)
    result = fixed_pc_block_cv(
        vectors,
        labels,
        groups,
        pc_candidates=[1, 3],
        random_state=9,
    )

    assert len(seen_group_values) == 5
    assert all(len(values) == 4 for values in seen_group_values)
    assert result.fold_scores.shape[0] == 10
    assert result.predictions.shape[0] == vectors.shape[0] * 2


def test_nested_pc_cv_returns_exactly_one_outer_prediction_per_trial():
    vectors, labels, groups = _decodable_grouped_data()
    candidates = [1, 2, 5, 10]
    result = nested_pc_block_cv(
        vectors,
        labels,
        groups,
        pc_candidates=candidates,
        random_state=11,
    )

    assert result.fold_scores.shape[0] == 5
    assert result.predictions.shape[0] == vectors.shape[0]
    assert result.predictions["trial_index"].is_unique
    assert set(result.fold_scores["selected_n_components"]).issubset(candidates)
    assert result.pooled_balanced_accuracy > 0.85
    assert result.mean_block_balanced_accuracy > 0.85
