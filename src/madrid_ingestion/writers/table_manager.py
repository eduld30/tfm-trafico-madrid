"""Registro seguro de tablas Delta externas en Unity Catalog."""

from __future__ import annotations

from typing import Any

from madrid_ingestion.core.exceptions import IngestionError


def quote_identifier(identifier: str) -> str:
    """Escapa un identificador para SQL de Spark."""
    return f"`{identifier.replace('`', '``')}`"


def quote_table_name(table_name: str) -> str:
    """Escapa por separado catálogo, esquema y tabla."""
    parts = table_name.split(".")
    if len(parts) != 3:
        raise IngestionError(f"Se esperaba <catalog>.<schema>.<table>: {table_name!r}.")
    return ".".join(quote_identifier(part) for part in parts)


class TableManager:
    """Gestiona metadatos de Unity Catalog sin borrar datos físicos."""

    def __init__(self, spark: Any) -> None:
        self.spark = spark

    def table_exists(self, table_name: str) -> bool:
        """Comprueba si una tabla está registrada."""
        return bool(self.spark.catalog.tableExists(table_name))

    def validate_table_location(self, table_name: str, expected_location: str) -> None:
        """Comprueba que una tabla existente apunta al LOCATION configurado."""
        detail = self.spark.sql(f"DESCRIBE DETAIL {quote_table_name(table_name)}")
        row = detail.select("location").first()
        if row is None:
            raise IngestionError(f"DESCRIBE DETAIL no devolvió datos para {table_name}.")
        actual_location = row["location"]
        if actual_location.rstrip("/") != expected_location.rstrip("/"):
            raise IngestionError(
                f"La tabla {table_name} está registrada en {actual_location!r}, "
                f"pero la configuración resuelve {expected_location!r}."
            )

    def ensure_external_table(self, table_name: str, location: str) -> None:
        """Crea el esquema y registra una tabla Delta externa si no existen."""
        if self.table_exists(table_name):
            self.validate_table_location(table_name, location)
            return
        catalog, schema, _ = table_name.split(".")
        self.spark.sql(
            f"CREATE SCHEMA IF NOT EXISTS "
            f"{quote_identifier(catalog)}.{quote_identifier(schema)}"
        )
        escaped_location = location.replace("'", "''")
        self.spark.sql(
            f"CREATE TABLE IF NOT EXISTS {quote_table_name(table_name)} "
            f"USING DELTA LOCATION '{escaped_location}'"
        )
