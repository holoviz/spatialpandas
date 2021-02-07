import numpy as np
import pytest
from hypothesis import given

from ..strategies import hyp_settings, st_multipolygon_array, st_polygon_array
from spatialpandas.geometry import MultiPolygonArray, PolygonArray


@pytest.mark.slow
@given(st_polygon_array())
@hyp_settings
def test_polygon_area(gp_polygon):
    polygons = PolygonArray.from_geopandas(gp_polygon)
    expected_area = gp_polygon.area
    area = polygons.area
    np.testing.assert_allclose(area, expected_area)


@pytest.mark.slow
@given(st_multipolygon_array())
@hyp_settings
def test_multipolygon_area(gp_multipolygon):
    multipolygons = MultiPolygonArray.from_geopandas(gp_multipolygon)
    expected_area = gp_multipolygon.area
    area = multipolygons.area
    np.testing.assert_allclose(area, expected_area)
