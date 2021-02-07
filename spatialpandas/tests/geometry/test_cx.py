import dask.dataframe as dd
import pytest
from hypothesis import given
from pandas.testing import assert_frame_equal, assert_series_equal

from .strategies import (
    hyp_settings,
    st_bounds,
    st_line_array,
    st_multiline_array,
    st_multipoint_array,
    st_multipolygon_array,
    st_point_array,
    st_polygon_array,
)
from spatialpandas import GeoDataFrame, GeoSeries
from spatialpandas.geometry import (
    LineArray,
    MultiLineArray,
    MultiPointArray,
    MultiPolygonArray,
    PointArray,
    PolygonArray,
)


def get_slices(v0, v1):
    return [slice(v0, v1), slice(None, v1), slice(v0, None), slice(None, None)]


@pytest.mark.slow
@given(st_point_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_point_cx_selection(gp_point, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = PointArray.from_geopandas(gp_point.cx[xslice, yslice])
            result = PointArray.from_geopandas(gp_point).cx[xslice, yslice]
            assert all(expected == result)


@pytest.mark.slow
@given(st_multipoint_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_selection(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = MultiPointArray.from_geopandas(gp_multipoint.cx[xslice, yslice])
            result = MultiPointArray.from_geopandas(gp_multipoint).cx[xslice, yslice]
            assert all(expected == result)


@pytest.mark.slow
@given(st_line_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_line_cx_selection(gp_line, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = LineArray.from_geopandas(gp_line.cx[xslice, yslice])
            result = LineArray.from_geopandas(gp_line).cx[xslice, yslice]
            assert all(expected == result)


@pytest.mark.slow
@given(st_multiline_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multiline_cx_selection(gp_multiline, rect):
    x0, y0, x1, y1 = rect
    for xslice in get_slices(x0, x1):
        for yslice in get_slices(y0, y1):
            expected = MultiLineArray.from_geopandas(gp_multiline.cx[xslice, yslice])
            result = MultiLineArray.from_geopandas(gp_multiline).cx[xslice, yslice]
            assert all(expected == result)


@pytest.mark.slow
@given(
    st_polygon_array(min_size=1, geoseries=True),
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


@pytest.mark.slow
@given(
    st_multipolygon_array(min_size=1, geoseries=True),
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


@given(st_multipoint_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_series_selection(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    expected = GeoSeries(gp_multipoint.cx[x0:x1, y0:y1], dtype='multipoint')

    sp_multipoint = GeoSeries(gp_multipoint)
    result = sp_multipoint.cx[x0:x1, y0:y1]
    assert_series_equal(expected, result, obj='GeoSeries')


@pytest.mark.slow
@given(st_multipoint_array(min_size=6, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_series_selection_dask(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    expected = GeoSeries(gp_multipoint.cx[x0:x1, y0:y1], dtype='multipoint')

    sp_multipoint = dd.from_pandas(GeoSeries(gp_multipoint), npartitions=3)
    result = sp_multipoint.cx[x0:x1, y0:y1].compute()
    assert_series_equal(expected, result, obj='GeoSeries')


@given(st_multipoint_array(min_size=1, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_frame_selection(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    expected = GeoDataFrame(
        GeoSeries(gp_multipoint.cx[x0:x1, y0:y1], dtype='multipoint')
    )

    sp_multipoint = GeoSeries(gp_multipoint).to_frame()
    result = sp_multipoint.cx[x0:x1, y0:y1]
    assert_frame_equal(expected, result, obj='GeoDataFrame')


@pytest.mark.slow
@given(st_multipoint_array(min_size=6, geoseries=True), st_bounds(orient=True))
@hyp_settings
def test_multipoint_cx_frame_selection_dask(gp_multipoint, rect):
    x0, y0, x1, y1 = rect
    expected = GeoDataFrame(
        GeoSeries(gp_multipoint.cx[x0:x1, y0:y1], dtype='multipoint')
    )

    sp_multipoint = dd.from_pandas(GeoSeries(gp_multipoint).to_frame(), npartitions=3)
    result = sp_multipoint.cx[x0:x1, y0:y1].compute()
    assert_frame_equal(expected, result, obj='GeoDataFrame')
