import ast
from pathlib import Path

import pytest

COMMAND_SEPARATOR = "# COMMAND ----------"
NOTEBOOK_DIR = Path(__file__).parents[2] / "notebooks"


@pytest.mark.parametrize(
    "notebook_name",
    ["eda_ml_gold.py", "run_ml_preprocessing.py"],
)
def test_ml_markdown_cells_do_not_swallow_python(notebook_name: str) -> None:
    source = (NOTEBOOK_DIR / notebook_name).read_text(encoding="utf-8")


    for cell_number, cell in enumerate(source.split(COMMAND_SEPARATOR), start=1):
        lines = [line for line in cell.splitlines() if line.strip()]
        if not lines or lines[0] != "# MAGIC %md":
            continue

        non_magic_lines = [line for line in lines if not line.startswith("# MAGIC")]
        assert non_magic_lines == [], (
            f"Databricks interpretará Python como Markdown en la celda {cell_number}: "
            f"{non_magic_lines[0]}"
        )


def test_preprocessing_notebook_uses_uc_volume_for_mlflow_spark_models() -> None:
    source = (NOTEBOOK_DIR / "run_ml_preprocessing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    spark_model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"log_model", "load_model"}
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "spark"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "mlflow"
    ]

    assert {call.func.attr for call in spark_model_calls} == {"log_model", "load_model"}
    for call in spark_model_calls:
        dfs_tmpdir = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "dfs_tmpdir"),
            None,
        )
        assert isinstance(dfs_tmpdir, ast.Name)
        assert dfs_tmpdir.id == "mlflow_dfs_tmp"


def test_preprocessing_notebook_creates_atomically_owned_mlflow_run() -> None:
    source = (NOTEBOOK_DIR / "run_ml_preprocessing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    mlflow_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "mlflow"
    ]

    assert [call for call in mlflow_calls if call.func.attr == "set_experiment"] == []
    start_runs = [call for call in mlflow_calls if call.func.attr == "start_run"]
    assert len(start_runs) == 1
    keywords = {keyword.arg: keyword.value for keyword in start_runs[0].keywords}
    assert isinstance(keywords["experiment_id"], ast.Name)
    assert keywords["experiment_id"].id == "mlflow_experiment_id"
    assert isinstance(keywords["run_name"], ast.JoinedStr)
    tags = keywords["tags"]
    assert isinstance(tags, ast.Dict)
    tag_values = {
        key.value: value
        for key, value in zip(tags.keys, tags.values, strict=True)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    owner = tag_values["madrid_ml.controller_token"]
    assert isinstance(owner, ast.Name)
    assert owner.id == "controller_token"


def test_preprocessing_notebook_result_exposes_artifact_proof() -> None:
    source = (NOTEBOOK_DIR / "run_ml_preprocessing.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    result_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        )
    ]

    assert len(result_assignments) == 1
    result = result_assignments[0].value
    assert isinstance(result, ast.Dict)
    result_keys = {
        key.value
        for key in result.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "controller_token",
        "mlflow_experiment_id",
        "package_version",
        "python_version",
        "pipeline_stage_count",
        "spark_version",
        "reload_parity",
    } <= result_keys
