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
