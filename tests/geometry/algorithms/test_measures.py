import numpy as np
from hypothesis import given
from spatialpandas.geometry import MultiPolygonArray, PolygonArray
from tests.geometry.strategies import (
    st_multipolygon_array, hyp_settings, st_polygon_array
)


@given(st_polygon_array())
@hyp_settings
def test_polygon_area(gp_polygon):
    polygons = PolygonArray.from_geopandas(gp_polygon)
    expected_area = gp_polygon.area
    area = polygons.area
    np.testing.assert_allclose(area, expected_area)


@given(st_multipolygon_array())
@hyp_settings
def test_multipolygon_area(gp_multipolygon):
    multipolygons = MultiPolygonArray.from_geopandas(gp_multipolygon)
    expected_area = gp_multipolygon.area
    area = multipolygons.area
    np.testing.assert_allclose(area, expected_area)
