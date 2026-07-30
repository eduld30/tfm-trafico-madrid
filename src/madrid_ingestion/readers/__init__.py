"""Registro explícito de lectores Auto Loader."""

from madrid_ingestion.core.exceptions import UnsupportedFormatError
from madrid_ingestion.readers.csv_reader import CsvAutoLoaderReader
from madrid_ingestion.readers.xml_reader import XmlAutoLoaderReader

READERS = {
    "csv": CsvAutoLoaderReader,
    "xml": XmlAutoLoaderReader,
}


def get_reader(file_format: str) -> CsvAutoLoaderReader | XmlAutoLoaderReader:
    """Crea el lector correspondiente al formato configurado."""
    try:
        return READERS[file_format]()
    except KeyError as exc:
        raise UnsupportedFormatError(f"Formato no soportado: {file_format!r}.") from exc


__all__ = ["READERS", "get_reader"]
