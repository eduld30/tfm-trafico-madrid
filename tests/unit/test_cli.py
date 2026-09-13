from pathlib import Path

from madrid_ingestion.cli import main

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "conf"


def test_validate_config_command(capsys):
    result = main(
        ["validate-config", "--env", "dev", "--config-root", str(CONFIG_ROOT)]
    )
    assert result == 0
    assert "datasets=14" in capsys.readouterr().out


def test_list_datasets_command(capsys):
    result = main(
        ["list-datasets", "--source", "trafico", "--config-root", str(CONFIG_ROOT)]
    )
    assert result == 0
    assert "trafico.trafico_nrt" in capsys.readouterr().out


def test_invalid_environment_returns_non_zero():
    result = main(
        ["validate-config", "--env", "unknown", "--config-root", str(CONFIG_ROOT)]
    )
    assert result == 2
