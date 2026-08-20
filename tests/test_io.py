import numpy as np
import pytest
from scipy.io import savemat

from kunz_speech_geometry.io import inspect_mat_file, load_interleaved_mat


def test_mat_audit_lists_shapes_without_loading_adapter(tmp_path):
    path = tmp_path / "tiny.mat"
    savemat(path, {"rates": [[1.0, 2.0], [3.0, 4.0]], "labels": [1, 2]})
    audit = inspect_mat_file(path)
    assert set(audit["name"]) == {"rates", "labels"}
    rate_shape = audit.loc[audit["name"] == "rates", "shape_on_disk"].iloc[0]
    assert rate_shape == (2, 2)


def _write_interleaved_fixture(path, *, participant="t15"):
    n_time, n_channels = 90, 4
    features = np.arange(n_time * n_channels).reshape(n_time, n_channels)
    channel_sets = np.empty(2, dtype=object)
    channel_sets[0] = np.array([1, 2])
    channel_sets[1] = np.array([3, 4])
    if participant == "t12":
        cues = np.array(
            [["DO_NOTHING$0"], ["ban$0"], ["ban\\n(attempted)$1"]], dtype=object
        )
        trial_cues = np.array([[2], [3]])
        go_epochs = np.array([[31, 40], [51, 60]])
        delay_epochs = np.array([[21, 30], [41, 50]])
    else:
        cues = np.array(
            [
                ["DO_NOTHING"],
                ["ban (attempted)"],
                ["ban (imaginedlistening)"],
                ["ban (passivelistening)"],
            ],
            dtype=object,
        )
        trial_cues = np.array([[1], [2], [4]])
        go_epochs = np.array([[11, 20], [31, 40], [61, 70]])
        delay_epochs = go_epochs - 5
    savemat(
        path,
        {
            "binnedTX": features,
            "spikePow": features.astype(float),
            "blockNum": np.ones((n_time, 1)),
            "goTrialEpochs": go_epochs,
            "delayTrialEpochs": delay_epochs,
            "trialCues": trial_cues,
            "cueList": cues,
            "chanSets": channel_sets,
            "chanSetNames": np.array([["i6v", "s6v"]], dtype=object),
            "binSize": np.array([[10]]),
        },
    )
    return features


def test_interleaved_adapter_maps_t15_conditions_and_channels(tmp_path):
    path = tmp_path / "t15.2024.06.14_interleavedVerbalBehaviors_raw.mat"
    features = _write_interleaved_fixture(path)

    dataset = load_interleaved_mat(
        path, channel_sets="i6v", epoch_ms=(-20, 30), block_center=False
    )

    assert dataset.rates.shape == (2, 5, 2)
    assert dataset.time_ms.tolist() == [-20, -10, 0, 10, 20]
    assert dataset.trials["condition"].tolist() == [
        "attempted_vocalized",
        "passive_listening",
    ]
    assert dataset.trials["word"].tolist() == ["ban", "ban"]
    assert dataset.channels["channel_id"].tolist() == ["ch001", "ch002"]
    np.testing.assert_array_equal(dataset.rates[0, 2], features[30, :2])


def test_t12_passive_uses_delay_start_but_attempted_uses_go_start(tmp_path):
    path = tmp_path / "t12.2024.04.11_interleavedVerbalBehaviors_raw.mat"
    features = _write_interleaved_fixture(path, participant="t12")

    dataset = load_interleaved_mat(path, epoch_ms=(0, 20), block_center=False)

    assert dataset.trials["alignment_event_raw"].tolist() == ["delay_start", "go_start"]
    np.testing.assert_array_equal(dataset.rates[0, 0], features[20])
    np.testing.assert_array_equal(dataset.rates[1, 0], features[50])


def test_interleaved_adapter_rejects_unknown_channel_set(tmp_path):
    path = tmp_path / "t15.2024.06.14_interleavedVerbalBehaviors_raw.mat"
    _write_interleaved_fixture(path)

    with pytest.raises(ValueError, match="Unknown channel sets"):
        load_interleaved_mat(path, channel_sets="not-an-array")
