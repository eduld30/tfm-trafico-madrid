from pathlib import Path

COMMAND_SEPARATOR = "# COMMAND ----------"
NOTEBOOK_PATH = Path(__file__).parents[2] / "notebooks" / "eda_ml_gold.py"


def test_eda_markdown_cells_do_not_swallow_python() -> None:
    source = NOTEBOOK_PATH.read_text(encoding="utf-8")

    for cell_number, cell in enumerate(source.split(COMMAND_SEPARATOR), start=1):
        lines = [line for line in cell.splitlines() if line.strip()]
        if not lines or lines[0] != "# MAGIC %md":
            continue

        non_magic_lines = [line for line in lines if not line.startswith("# MAGIC")]
        assert non_magic_lines == [], (
            f"Databricks interpretará Python como Markdown en la celda {cell_number}: "
            f"{non_magic_lines[0]}"
        )
