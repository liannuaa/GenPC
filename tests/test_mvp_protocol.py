import numpy as np

from src.mvp_protocol import MVP_CATEGORIES, build_test_records, case_key
from src.offline_metrics import fscore_from_squared_distances


def test_mvp_test_records_preserve_h5_view_contract():
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    records = build_test_records(partial_count=8, complete_count=2, labels=labels, start=4, stop=7)
    assert [record["case_key"] for record in records] == ["mvp_test_00004", "mvp_test_00005", "mvp_test_00006"]
    assert [record["complete_index"] for record in records] == [1, 1, 1]
    assert [record["view_index"] for record in records] == [0, 1, 2]
    assert records[0]["category"] == MVP_CATEGORIES[4]
    assert case_key(41600) == "mvp_test_41600"


def test_fscore_uses_squared_distances_for_a_euclidean_threshold():
    precision, recall, fscore = fscore_from_squared_distances(
        np.array([0.0, 0.0001, 0.00010001]),
        np.array([0.0, 0.0001]),
        threshold=0.01,
    )
    assert precision == 2.0 / 3.0
    assert recall == 1.0
    assert fscore == 0.8
