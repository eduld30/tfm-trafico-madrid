"""Excepciones de dominio del motor de ingesta."""


class IngestionError(Exception):
    """Error base del paquete."""


class ConfigurationError(IngestionError):
    """La configuración no existe o incumple el contrato."""


class DatasetNotFoundError(ConfigurationError):
    """La fuente o el dataset solicitado no existe."""


class UnsupportedFormatError(IngestionError):
    """El formato de entrada no dispone de lector."""


class UnsupportedTransformationError(IngestionError):
    """La transformación declarada no está implementada."""


class WriteStrategyError(IngestionError):
    """La estrategia de escritura no es válida o no puede ejecutarse."""
