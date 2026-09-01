import pytest

from madrid_ml.evaluation import global_binary_metrics, segmented_binary_metrics


def test_global_metrics_handle_tied_scores_deterministically(spark):
    scored = spark.createDataFrame(
        [(1, 0.9), (0, 0.9), (1, 0.2), (0, 0.1)],
        "target_accident_next_hour int, score double",
    )

    metrics = global_binary_metrics(scored)

    assert metrics["rows"] == 4
    assert metrics["positives"] == 2
    assert metrics["average_precision"] == pytest.approx(7 / 12)
    assert metrics["brier_score"] == pytest.approx(0.3675)


def test_segment_metrics_keep_single_class_segments_as_not_evaluable(spark):
    scored = spark.createDataFrame(
        [(1, 8, 1, 0.8), (1, 8, 1, 0.7), (2, 8, 1, 0.9), (2, 8, 0, 0.1)],
        "cod_distrito int, hora_dia int, target_accident_next_hour int, score double",
    )

    rows = {
        row.cod_distrito: row for row in segmented_binary_metrics(scored).collect()
    }

    assert rows[1].rows == 2 and rows[1].positives == 2
    assert rows[1].average_precision is None and rows[1].roc_auc is None
    assert rows[2].average_precision == pytest.approx(1.0)
