from pathlib import Path

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.runner import build_run_context

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "conf"


def test_bronze_context_resolves_all_physical_paths():
    loader = ConfigLoader(CONFIG_ROOT)
    environment = loader.load_environment("dev")
    _, dataset = loader.load_dataset("trafico", "trafico_nrt")
    context = build_run_context(environment, dataset, "bronze", "trafico", "run-1")
    assert context.table == "dev_bronze.trafico.trafico_nrt"
    assert context.source_path.startswith("abfss://landing@")
    assert context.target_path.startswith("abfss://bronze@")
    assert context.checkpoint_path == f"{context.target_path}_checkpoint/"
    assert context.schema_path == context.checkpoint_path
    assert context.source_table is None


def test_silver_reads_table_and_has_no_checkpoint():
    loader = ConfigLoader(CONFIG_ROOT)
    environment = loader.load_environment("dev")
    _, dataset = loader.load_dataset("trafico", "dim_trafico")
    context = build_run_context(environment, dataset, "silver", "trafico", "run-2")
    assert context.source_table == "dev_bronze.trafico.dim_trafico"
    assert context.table == "dev_silver.trafico.dim_trafico"
    assert context.target_path.startswith("abfss://silver@")
    assert context.checkpoint_path is None
    assert context.schema_path is None
