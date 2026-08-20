import numpy as np

from kunz_speech_geometry import make_synthetic_dataset


def test_synthetic_dataset_obeys_contract():
    dataset = make_synthetic_dataset(
        n_channels=16,
        n_trials_per_word_condition=3,
        random_state=7,
    )
    dataset.validate()
    assert dataset.rates.shape[0] == 2 * 7 * 3
    assert dataset.rates.shape[2] == 16
    assert set(dataset.trials["condition"]) == {
        "attempted_vocalized",
        "passive_listening",
    }
    assert np.all(np.diff(dataset.time_ms) > 0)


def test_condition_subset_keeps_metadata_aligned():
    dataset = make_synthetic_dataset(n_channels=12, n_trials_per_word_condition=2)
    subset = dataset.subset_conditions(["passive_listening"])
    assert subset.rates.shape[0] == len(subset.trials)
    assert subset.trials["condition"].unique().tolist() == ["passive_listening"]
