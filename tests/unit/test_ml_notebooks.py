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
