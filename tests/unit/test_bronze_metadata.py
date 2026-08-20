import re

from madrid_ingestion.bronze.metadata import FILE_DATE_PATTERN


def test_file_date_pattern_extracts_landing_partition_date():
    source_file = (
        "abfss://landing@account.dfs.core.windows.net/"
        "trafico/dim_distritos/2026/08/20/dim_distritos.csv"
    )

    match = re.search(FILE_DATE_PATTERN, source_file)

    assert match is not None
    assert match.group(1) == "2026/08/20"