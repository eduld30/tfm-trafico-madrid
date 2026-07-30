"""Ejecución acotada de Structured Streaming con trigger availableNow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def write_available_now(
    df: Any,
    target_path: str,
    checkpoint_path: str,
    partition_by: Sequence[str] = (),
) -> None:
    """Escribe los ficheros disponibles y espera la finalización del job."""
    writer = (
        df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", checkpoint_path)
    )
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    query = writer.trigger(availableNow=True).start(target_path)
    query.awaitTermination()
