"""CLI fina basada en argparse."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.observability.logger import configure_logging, get_logger
from madrid_ingestion.runner import run_dataset


def build_parser() -> argparse.ArgumentParser:
    """Construye el contrato de línea de comandos."""
    parser = argparse.ArgumentParser(prog="madrid-ingestion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Ejecuta datasets secuencialmente")
    run_parser.add_argument("--env", required=True)
    run_parser.add_argument("--layer", choices=("bronze", "silver"), required=True)
    run_parser.add_argument("--source")
    run_parser.add_argument("--dataset")
    run_parser.add_argument("--all", action="store_true", dest="run_all")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--config-root", type=Path)
    run_parser.add_argument("--log-level")

    validate_parser = subparsers.add_parser(
        "validate-config", help="Valida entorno y todas las fuentes"
    )
    validate_parser.add_argument("--env", required=True)
    validate_parser.add_argument("--config-root", type=Path)

    list_parser = subparsers.add_parser(
        "list-datasets", help="Lista los datasets de una fuente"
    )
    list_parser.add_argument("--source", required=True)
    list_parser.add_argument("--config-root", type=Path)
    return parser


def _selected_datasets(args: argparse.Namespace, loader: ConfigLoader) -> list[tuple[str, str]]:
    if args.run_all:
        if args.source or args.dataset:
            raise IngestionError("--all no se puede combinar con --source o --dataset.")
        selections: list[tuple[str, str]] = []
        for source_name in loader.source_names():
            source_config = loader.load_source(source_name)
            for dataset in source_config.datasets:
                layer_config = getattr(dataset, args.layer)
                if source_config.enabled and dataset.enabled and layer_config.enabled:
                    selections.append((source_name, dataset.name))
        return selections
    if not args.source:
        raise IngestionError("--source es obligatorio salvo cuando se usa --all.")
    source_config = loader.load_source(args.source)
    if args.dataset:
        return [(source_config.source, args.dataset)]
    return [
        (source_config.source, dataset.name)
        for dataset in source_config.datasets
        if source_config.enabled
        and dataset.enabled
        and getattr(dataset, args.layer).enabled
    ]


def _run_command(args: argparse.Namespace) -> int:
    loader = ConfigLoader(args.config_root)
    selections = _selected_datasets(args, loader)
    if not selections:
        raise IngestionError("La selección no contiene datasets habilitados.")
    results = [
        run_dataset(
            environment=args.env,
            layer=args.layer,
            source=source,
            dataset=dataset,
            run_id=args.run_id,
            config_root=args.config_root,
            log_level=args.log_level,
        )
        for source, dataset in selections
    ]
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    loader = ConfigLoader(args.config_root)
    environment, sources = loader.validate_all(args.env)
    dataset_count = sum(len(source.datasets) for source in sources)
    print(
        f"Configuración válida: entorno={environment.environment}, "
        f"fuentes={len(sources)}, datasets={dataset_count}."
    )
    return 0


def _list_command(args: argparse.Namespace) -> int:
    source = ConfigLoader(args.config_root).load_source(args.source)
    for dataset in source.datasets:
        bronze = "enabled" if dataset.bronze.enabled else "disabled"
        silver = "enabled" if dataset.silver.enabled else "disabled"
        state = "enabled" if dataset.enabled else "disabled"
        print(f"{source.source}.{dataset.name} [{state}] bronze={bronze} silver={silver}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la CLI y convierte errores de dominio en código distinto de cero."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run_command(args)
        if args.command == "validate-config":
            return _validate_command(args)
        return _list_command(args)
    except IngestionError as exc:
        configure_logging("ERROR")
        get_logger("cli").error("%s", exc)
        return 2
    except KeyboardInterrupt:
        print("Ejecución interrumpida.", file=sys.stderr)
        return 130
