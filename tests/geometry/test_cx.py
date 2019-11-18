from hypothesis import given

from spatialpandas.geometry import (
    MultiPointArray, LineArray, MultiLineArray, PolygonArray, MultiPolygonArray
)
from tests.geometry.strategies import (
    st_multipoint_array, st_bounds, st_line_array, st_multiline_array,
    st_polygons_array, st_multipolygons_array, hyp_settings
)


def get_slices(v0, v1):
    return [slice(v0, v1), slice(None, v1), slice(v0, None), slice(None, None)]


@given(st_multipoint_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_selection(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = MultiPointArray.from_geopandas(gp_multipoint.cx[xslice, yslice])
            result = MultiPointArray.from_geopandas(gp_multipoint).cx[xslice, yslice]
            assert all(expected == result)


@given(st_line_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_line_cx_selection(gp_line, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = LineArray.from_geopandas(gp_line.cx[xslice, yslice])
            result = LineArray.from_geopandas(gp_line).cx[xslice, yslice]
            assert all(expected == result)


@given(st_multiline_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multiline_cx_selection(gp_multiline, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = MultiLineArray.from_geopandas(gp_multiline.cx[xslice, yslice])
            result = MultiLineArray.from_geopandas(gp_multiline).cx[xslice, yslice]
            assert all(expected == result)


@given(
    st_polygons_array(min_size=1, geoseries=True),
    st_bounds(
        x_min=-150, x_max=150, y_min=-150, y_max=150, orient=True
    )
)
@hyp_settings
def test_polygon_cx_selection(gp_polygon, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = PolygonArray.from_geopandas(gp_polygon.cx[xslice, yslice])
            result = PolygonArray.from_geopandas(gp_polygon).cx[xslice, yslice]
            assert all(expected == result)


@given(
    st_multipolygons_array(min_size=1, geoseries=True),
    st_bounds(
        x_min=-150, x_max=150, y_min=-150, y_max=150, orient=True
    )
)
@hyp_settings
def test_multipolygon_cx_selection(gp_multipolygon, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = MultiPolygonArray.from_geopandas(
                gp_multipolygon.cx[xslice, yslice]
            )
            result = MultiPolygonArray.from_geopandas(
                gp_multipolygon
            ).cx[xslice, yslice]
            assert all(expected == result)
