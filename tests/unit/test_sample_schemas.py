import csv
from pathlib import Path
from xml.etree import ElementTree

from madrid_ingestion.config.loader import ConfigLoader
from madrid_ingestion.core.columns import normalize_column_name

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "conf"
SAMPLES_ROOT = ROOT / "samples"

SAMPLES = {
    ("accidentes", "accidentes_historico"): SAMPLES_ROOT
    / "accidentes"
    / "accidentes.csv",
    ("calair", "calair_historico"): SAMPLES_ROOT
    / "calair"
    / "calair_historico.csv",
    ("calair", "calair_nrt"): SAMPLES_ROOT / "calair" / "calair_nrt.csv",
    ("calair", "dim_calair"): SAMPLES_ROOT / "calair" / "dim_calair.csv",
    ("eventos", "eventos_culturales"): SAMPLES_ROOT
    / "eventos"
    / "eventos_culturales.csv",
    ("meteo", "meteo_historico"): SAMPLES_ROOT / "meteo" / "meteo_historico.csv",
    ("meteo", "meteo_nrt"): SAMPLES_ROOT / "meteo" / "meteo_nrt.csv",
    ("meteo", "dim_meteo"): SAMPLES_ROOT / "meteo" / "dim_meteo.csv",
    ("trafico", "dim_distritos"): SAMPLES_ROOT / "trafico" / "dim_distritos.csv",
    ("trafico", "dim_trafico"): SAMPLES_ROOT / "trafico" / "dim_trafico.csv",
    ("trafico", "trafico_historico"): SAMPLES_ROOT
    / "trafico"
    / "trafico_historico.csv",
    ("trafico", "trafico_nrt"): SAMPLES_ROOT / "trafico" / "trafico_nrt.xml",
}

TECHNICAL_COLUMNS = {
    "_ingestion_timestamp",
    "_ingestion_run_id",
    "_source_file",
    "_source_file_modification_time",
}


def _source_columns(path, dataset):
    if path.suffix == ".xml":
        root = ElementTree.parse(path).getroot()
        record_path = dataset.bronze.reader_options["record_path"]
        parent_columns = dataset.bronze.reader_options["parent_columns"].split(",")
        record_columns = {
            child.tag
            for record in root.findall(record_path)
            for child in record
        }
        return [*parent_columns, *sorted(record_columns)]

    encoding = dataset.bronze.reader_options["encoding"]
    with path.open(encoding=encoding, newline="") as sample:
        return next(csv.reader(sample, delimiter=";", quotechar='"'))


def _bronze_columns(columns, dataset):
    result = columns
    for transformation in dataset.bronze.transformations:
        if transformation.type == "normalize_column_names":
            result = [normalize_column_name(column) for column in result]
        else:
            result = [
                transformation.columns.get(column, column) for column in result
            ]
    return [*result, *TECHNICAL_COLUMNS]


def _assert_present(columns, required, dataset_name, transformation):
    missing = sorted(set(required).difference(columns))
    assert not missing, (
        f"{dataset_name}.{transformation} referencia columnas ausentes: {missing}"
    )


def _silver_columns(columns, dataset):
    result = list(columns)
    for transformation in dataset.silver.transformations:
        name = transformation.type
        if name in {"trim", "empty_to_null", "regex_replace"}:
            _assert_present(result, transformation.columns, dataset.name, name)
        elif name == "cast":
            _assert_present(result, transformation.columns, dataset.name, name)
        elif name == "drop":
            _assert_present(result, transformation.columns, dataset.name, name)
            result = [column for column in result if column not in transformation.columns]
        elif name == "parse_timestamp":
            sources = transformation.source_columns or [
                transformation.source_column
            ]
            _assert_present(result, sources, dataset.name, name)
            if transformation.target_column not in result:
                result.append(transformation.target_column)
        elif name == "parse_date":
            _assert_present(
                result, [transformation.source_column], dataset.name, name
            )
            if transformation.target_column not in result:
                result.append(transformation.target_column)
        elif name == "hourly_wide_to_long":
            hours = transformation.hours or 24
            wide = [
                f"{prefix}{hour:02d}"
                for hour in range(1, hours + 1)
                for prefix in (
                    transformation.value_prefix,
                    transformation.validity_prefix,
                )
            ]
            _assert_present(
                result,
                [
                    transformation.year_column,
                    transformation.month_column,
                    transformation.day_column,
                    *wide,
                ],
                dataset.name,
                name,
            )
            result = [column for column in result if column not in wide]
            result.extend(
                [
                    transformation.value_column,
                    transformation.validity_column,
                    transformation.timestamp_column,
                ]
            )
        elif name == "deduplicate":
            _assert_present(
                result,
                [
                    *transformation.keys,
                    *(item.column for item in transformation.order_by),
                ],
                dataset.name,
                name,
            )
    return result


def test_all_available_samples_have_compatible_silver_configuration():
    loader = ConfigLoader(CONFIG_ROOT)
    assert len(SAMPLES) == 12

    for (source_name, dataset_name), sample_path in SAMPLES.items():
        _, dataset = loader.load_dataset(source_name, dataset_name)
        assert sample_path.is_file()
        assert dataset.silver.enabled
        source_columns = _source_columns(sample_path, dataset)
        bronze_columns = _bronze_columns(source_columns, dataset)
        silver_columns = _silver_columns(bronze_columns, dataset)
        assert silver_columns
        if dataset.silver.write_strategy == "merge":
            _assert_present(
                silver_columns,
                dataset.silver.business_keys,
                dataset.name,
                "business_keys",
            )
