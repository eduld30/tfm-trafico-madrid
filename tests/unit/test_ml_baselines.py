import pytest

from madrid_ml.baselines import fit_baselines, score_baselines


def test_baselines_fit_train_frequencies_and_apply_fallback_chain(spark):
    train = spark.createDataFrame(
        [
            (1, 8, 1, 1),
            (1, 8, 1, 0),
            (1, 9, 1, 0),
            (2, 8, 1, 1),
        ],
        "cod_distrito int, hora_dia int, dia_semana int, "
        "target_accident_next_hour int",
    )
    scoring = spark.createDataFrame(
        [(1, 8, 1, 0), (1, 23, 7, 0), (3, 23, 7, 0)],
        train.schema,
    )

    state = fit_baselines(train)
    scored = score_baselines(state, scoring)
    scores = {
        (row.cod_distrito, row.hora_dia, row.dia_semana): row.score
        for row in scored["district_hour_day"].collect()
    }

    assert state.global_prevalence == 0.5
    assert scores[(1, 8, 1)] == 0.5
    assert scores[(1, 23, 7)] == pytest.approx(1 / 3)
    assert scores[(3, 23, 7)] == 0.5
