import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from madrid_ml.registry import ModelPromotionError, promote_model_run


class FakeClient:
    def __init__(self, tags):
        self.tags = tags
        self.alias_calls = []
        self.version_tag_calls = []

    def get_run(self, run_id):
        assert run_id == "model-run"
        return SimpleNamespace(data=SimpleNamespace(tags=self.tags))

    def set_registered_model_alias(self, *args):
        self.alias_calls.append(args)

    def set_model_version_tag(self, *args):
        self.version_tag_calls.append(args)


def _fake_mlflow(client):
    return SimpleNamespace(
        set_registry_uri=Mock(),
        register_model=Mock(return_value=SimpleNamespace(version="7")),
        MlflowClient=Mock(return_value=client),
    )


def test_promote_model_run_registers_classifier_and_moves_alias(monkeypatch):
    client = FakeClient(
        {
            "madrid_ml.comparator": "logistic_regression",
            "mlflow.parentRunId": "parent-run",
        }
    )
    mlflow = _fake_mlflow(client)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)

    result = promote_model_run(
        model_run_id="model-run",
        registered_model_name="dev_gold.ml.accident_risk_model",
    )

    mlflow.register_model.assert_called_once_with(
        "runs:/model-run/classifier",
        "dev_gold.ml.accident_risk_model",
        await_registration_for=300,
    )
    assert result.model_version == "7"
    assert client.alias_calls == [
        ("dev_gold.ml.accident_risk_model", "Champion", "7")
    ]


def test_promote_model_run_rejects_child_without_parent(monkeypatch):
    client = FakeClient(
        {
            "madrid_ml.comparator": "custom_model",
        }
    )
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(client))

    with pytest.raises(ModelPromotionError, match="child run with a parent"):
        promote_model_run(
            model_run_id="model-run",
            registered_model_name="dev_gold.ml.accident_risk_model",
        )
