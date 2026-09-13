import pytest

from madrid_ingestion.core.exceptions import IngestionError
from madrid_ingestion.geospatial.districts import (
    assign_district,
    load_district_boundaries,
)


def test_packaged_kml_contains_the_21_valid_districts():
    boundaries = load_district_boundaries()

    assert [boundary.code for boundary in boundaries] == list(range(1, 22))
    assert len({boundary.name for boundary in boundaries}) == 21
    assert all(boundary.polygon.is_valid for boundary in boundaries)


def test_each_polygon_representative_point_maps_to_itself():
    boundaries = load_district_boundaries()

    for boundary in boundaries:
        point = boundary.polygon.representative_point()
        assigned = assign_district(point.x, point.y, boundaries)
        assert assigned.code == boundary.code


def test_point_outside_madrid_is_not_approximated():
    with pytest.raises(IngestionError, match="Ningún distrito"):
        assign_district(0.0, 0.0, load_district_boundaries())
