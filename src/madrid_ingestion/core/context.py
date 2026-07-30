"""Objetos inmutables que describen y resumen una ejecución."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunContext:
    """Contexto resuelto para una única ejecución dataset/capa."""

    environment: str
    layer: str
    source: str
    dataset: str
    run_id: str
    catalog: str
    schema: str
    table: str
    source_table: str | None
    source_path: str
    target_path: str
    checkpoint_path: str | None
    schema_path: str | None


@dataclass(frozen=True)
class RunResult:
    """Resultado estable y serializable de una ejecución completa."""

    run_id: str
    environment: str
    layer: str
    source: str
    dataset: str
    target_table: str
    target_path: str
    status: str
