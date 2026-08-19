from madrid_ingestion.readers.csv_reader import CsvAutoLoaderReader
from madrid_ingestion.readers.xml_reader import XmlAutoLoaderReader


class FakeReader:
    def __init__(self):
        self.options = {}
        self.format_name = None
        self.loaded_path = None

    def format(self, value):
        self.format_name = value
        return self

    def option(self, key, value):
        self.options[key] = value
        return self

    def load(self, path):
        self.loaded_path = path
        return self


class FakeSpark:
    def __init__(self):
        self.readStream = FakeReader()


def test_csv_reader_builds_cloudfiles_stream():
    spark = FakeSpark()
    result = CsvAutoLoaderReader().read_stream(
        spark,
        "landing-path",
        "schema-path",
        {"delimiter": ";"},
        {"cloudFiles.schemaEvolutionMode": "addNewColumns"},
    )
    assert result.format_name == "cloudFiles"
    assert result.options["cloudFiles.format"] == "csv"
    assert result.options["delimiter"] == ";"
    assert result.loaded_path == "landing-path"


def test_xml_reader_translates_row_tag():
    spark = FakeSpark()
    result = XmlAutoLoaderReader().read_stream(
        spark,
        "landing-path",
        "schema-path",
        {"row_tag": "pm"},
        {},
    )
    assert result.options["cloudFiles.format"] == "xml"
    assert result.options["rowTag"] == "pm"
    assert "row_tag" not in result.options
