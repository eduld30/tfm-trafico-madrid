"""Asignación de estaciones a los distritos del KML oficial de Madrid."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from madrid_ingestion.core.exceptions import IngestionError

_KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
_KML_NS = {"kml": _KML_NAMESPACE}
_EXPECTED_DISTRICT_CODES = frozenset(range(1, 22))
_RESOURCE_PACKAGE = "madrid_ingestion.resources"
_RESOURCE_NAME = "distritos.kml"


@dataclass(frozen=True)
class DistrictBoundary:
    """Código, nombre y polígono de un distrito de Madrid."""

    code: int
    name: str
    polygon: Any


def _shapely_geometry() -> tuple[Any, Any]:
    try:
        from shapely.geometry import Point, Polygon
    except ImportError as exc:
        raise IngestionError(
            "La transformación assign_district requiere Shapely. "
            "Instale las dependencias del paquete madrid-ingestion."
        ) from exc
    return Point, Polygon


def _description_field(description: str, field: str) -> str:
    pattern = rf"<td>{re.escape(field)}</td>\s*<td>([^<]*)</td>"
    match = re.search(pattern, description)
    if match is None:
        raise IngestionError(f"El KML de distritos no contiene el campo {field!r}.")
    return match.group(1).strip()


def _parse_coordinates(raw_coordinates: str, district_code: int) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for raw_point in raw_coordinates.split():
        parts = raw_point.split(",")
        if len(parts) != 3:
            raise IngestionError(
                f"El distrito {district_code} contiene una coordenada KML inválida: {raw_point!r}."
            )
        try:
            longitude = float(parts[0])
            latitude = float(parts[1])
        except ValueError as exc:
            raise IngestionError(
                f"El distrito {district_code} contiene una coordenada no numérica: {raw_point!r}."
            ) from exc
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise IngestionError(f"El distrito {district_code} contiene una coordenada no finita.")
        coordinates.append((longitude, latitude))
    if len(coordinates) < 4:
        raise IngestionError(f"El distrito {district_code} necesita al menos cuatro coordenadas.")
    if coordinates[0] != coordinates[-1]:
        raise IngestionError(f"El contorno del distrito {district_code} no está cerrado.")
    return coordinates


def parse_district_boundaries(kml_text: str) -> tuple[DistrictBoundary, ...]:
    """Parsea y valida la estructura concreta del KML oficial del proyecto."""
    try:
        root = ET.fromstring(kml_text)
    except ET.ParseError as exc:
        raise IngestionError(f"El recurso de distritos no es un KML válido: {exc}.") from exc

    _, Polygon = _shapely_geometry()
    placemarks = root.findall(".//kml:Placemark", _KML_NS)
    if len(placemarks) != len(_EXPECTED_DISTRICT_CODES):
        raise IngestionError(
            f"El KML debe contener exactamente 21 distritos; se encontraron {len(placemarks)}."
        )

    boundaries: list[DistrictBoundary] = []
    for placemark in placemarks:
        description = placemark.findtext("kml:description", default="", namespaces=_KML_NS)
        try:
            district_code = int(_description_field(description, "COD_DIS"))
        except ValueError as exc:
            raise IngestionError("COD_DIS debe ser un número entero en el KML.") from exc
        district_name = _description_field(description, "NOMBRE")
        coordinate_nodes = placemark.findall(
            "kml:MultiGeometry/kml:LineString/kml:coordinates", _KML_NS
        )
        if len(coordinate_nodes) != 1 or not coordinate_nodes[0].text:
            raise IngestionError(
                f"El distrito {district_code} debe contener exactamente un LineString."
            )
        coordinates = _parse_coordinates(coordinate_nodes[0].text, district_code)
        polygon = Polygon(coordinates)
        if polygon.is_empty or not polygon.is_valid:
            raise IngestionError(
                f"El contorno del distrito {district_code} no forma un polígono válido."
            )
        boundaries.append(DistrictBoundary(district_code, district_name, polygon))

    codes = [boundary.code for boundary in boundaries]
    if len(codes) != len(set(codes)):
        raise IngestionError("El KML contiene códigos de distrito duplicados.")
    if set(codes) != _EXPECTED_DISTRICT_CODES:
        raise IngestionError("El KML debe contener los códigos de distrito del 1 al 21.")
    return tuple(sorted(boundaries, key=lambda boundary: boundary.code))


def load_district_boundaries() -> tuple[DistrictBoundary, ...]:
    """Carga los límites versionados dentro del paquete instalado."""
    resource = files(_RESOURCE_PACKAGE).joinpath(_RESOURCE_NAME)
    return parse_district_boundaries(resource.read_text(encoding="utf-8"))


def assign_district(
    longitude: float,
    latitude: float,
    boundaries: Sequence[DistrictBoundary],
) -> DistrictBoundary:
    """Obtiene el único distrito que cubre un punto; no aproxima por cercanía."""
    Point, _ = _shapely_geometry()
    point = Point(longitude, latitude)
    matches = [boundary for boundary in boundaries if boundary.polygon.covers(point)]
    if not matches:
        raise IngestionError(
            f"Ningún distrito contiene las coordenadas longitud={longitude}, latitud={latitude}."
        )
    if len(matches) > 1:
        codes = ", ".join(str(boundary.code) for boundary in matches)
        raise IngestionError(
            f"Las coordenadas están en una frontera compartida por varios distritos: {codes}."
        )
    return matches[0]


def enrich_stations_with_district(
    df: Any,
    *,
    station_key: str,
    longitude_column: str,
    latitude_column: str,
    district_code_column: str,
    district_name_column: str,
) -> Any:
    """Añade distrito a una dimensión pequeña de estaciones mediante Shapely."""
    required = {station_key, longitude_column, latitude_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise IngestionError("assign_district: no existen las columnas " + ", ".join(missing) + ".")
    output_columns = {district_code_column, district_name_column}
    collisions = sorted(output_columns.intersection(df.columns))
    if collisions:
        raise IngestionError(
            "assign_district no puede sobrescribir columnas: " + ", ".join(collisions)
        )
    if district_code_column == district_name_column:
        raise IngestionError("Las columnas de código y nombre de distrito deben ser distintas.")

    station_rows = df.select(station_key, longitude_column, latitude_column).collect()
    if not station_rows:
        raise IngestionError("assign_district: no hay estaciones que procesar.")

    boundaries = load_district_boundaries()
    assignments: list[tuple[Any, int, str]] = []
    seen_keys: set[Any] = set()
    for row in station_rows:
        key = row[station_key]
        longitude = row[longitude_column]
        latitude = row[latitude_column]
        if key is None or longitude is None or latitude is None:
            raise IngestionError(
                f"assign_district requiere código, longitud y latitud no nulos; registro={row!r}."
            )
        if key in seen_keys:
            raise IngestionError(f"assign_district: código de estación duplicado: {key!r}.")
        seen_keys.add(key)
        try:
            longitude_value = float(longitude)
            latitude_value = float(latitude)
        except (TypeError, ValueError) as exc:
            raise IngestionError(
                f"assign_district: coordenadas no numéricas para la estación {key!r}."
            ) from exc
        if not math.isfinite(longitude_value) or not math.isfinite(latitude_value):
            raise IngestionError(
                f"assign_district: coordenadas no finitas para la estación {key!r}."
            )
        try:
            district = assign_district(longitude_value, latitude_value, boundaries)
        except IngestionError as exc:
            raise IngestionError(
                f"No se pudo asignar distrito a la estación {key!r}: {exc}"
            ) from exc
        assignments.append((key, district.code, district.name))

    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    key_field = next(field for field in df.schema.fields if field.name == station_key)
    mapping_schema = StructType(
        [
            StructField(station_key, key_field.dataType, nullable=False),
            StructField(district_code_column, IntegerType(), nullable=False),
            StructField(district_name_column, StringType(), nullable=False),
        ]
    )
    mapping_df = df.sparkSession.createDataFrame(assignments, schema=mapping_schema)
    source_alias = df.alias("stations")
    mapping_alias = mapping_df.alias("districts")
    join_condition = F.col(f"stations.`{station_key}`") == F.col(f"districts.`{station_key}`")
    return source_alias.join(mapping_alias, join_condition, "left").select(
        "stations.*",
        F.col(f"districts.`{district_code_column}`").alias(district_code_column),
        F.col(f"districts.`{district_name_column}`").alias(district_name_column),
    )
