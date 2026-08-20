from scipy.io import savemat

from kunz_speech_geometry.io import inspect_mat_file


def test_mat_audit_lists_shapes_without_loading_adapter(tmp_path):
    path = tmp_path / "tiny.mat"
    savemat(path, {"rates": [[1.0, 2.0], [3.0, 4.0]], "labels": [1, 2]})
    audit = inspect_mat_file(path)
    assert set(audit["name"]) == {"rates", "labels"}
    rate_shape = audit.loc[audit["name"] == "rates", "shape_on_disk"].iloc[0]
    assert rate_shape == (2, 2)
