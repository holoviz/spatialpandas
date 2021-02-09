import numpy as np
import pandas as pd
import shapely.geometry as sg
from geopandas.array import from_shapely
from hypothesis import example, given
from hypothesis import strategies as st

from ..strategies import (
    hyp_settings,
    st_line_array,
    st_multipoint_array,
    st_multipolygon_array,
    st_point_array,
    st_polygon_array,
)
from spatialpandas import GeoSeries
from spatialpandas.geometry import (
    Line,
    MultiLine,
    MultiPoint,
    MultiPolygon,
    Point,
    PointArray,
    Polygon,
)


@given(st_point_array(), st_point_array(min_size=1, max_size=1))
@hyp_settings
def test_points_intersects_point(gp_points, gp_point):
    # Get scalar Point
    sg_point = gp_point[0]
    if len(gp_points) > 0:
        # Add gp_point to gp_points so we know something will intersect
        gp_points = pd.concat([pd.Series(gp_points), pd.Series(gp_point)]).array

    # Compute expected intersection
    expected = gp_points.intersects(sg_point)

    # Create spatialpandas PointArray
    point = Point.from_shapely(sg_point)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(point) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(point)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(point, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(point),
        pd.Series(expected, index=points_series.index)
    )


@given(
    st_point_array(min_size=6),
    st.integers(min_value=0, max_value=5),
    st_point_array(min_size=1, max_size=1)
)
@hyp_settings
def test_points_intersects_point_offset(gp_points, offset, gp_point):
    # Get scalar Point
    sg_point = gp_point[0]
    if len(gp_points) > 0:
        # Add gp_point to gp_points so we know something will intersect
        gp_points = pd.concat([pd.Series(gp_points), pd.Series(gp_point)]).array

    # Compute expected intersection
    expected = gp_points.intersects(sg_point)[offset:]

    # Create spatialpandas PointArray
    point = Point.from_shapely(sg_point)
    points = PointArray.from_geopandas(gp_points)[offset:]

    # Test PointArray.intersect
    result = points.intersects(point)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(point, inds)
    np.testing.assert_equal(result, np.flipud(expected))


@given(st_point_array(), st_multipoint_array(min_size=1, max_size=1))
@hyp_settings
def test_points_intersects_multipoint(gp_points, gp_multipoint):
    # Get scalar Point
    sg_multipoint = gp_multipoint[0]
    if len(gp_points) > 0:
        # Add gp_point to gp_multipoints so we know something will intersect
        gp_points = from_shapely(list(gp_points) + [gp_multipoint[0][-1]])

    # Compute expected intersection
    expected = gp_points.intersects(sg_multipoint)

    # Create spatialpandas PointArray
    multipoint = MultiPoint.from_shapely(sg_multipoint)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(multipoint) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(multipoint)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(multipoint, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(multipoint),
        pd.Series(expected, index=points_series.index)
    )


@given(st_point_array(), st_line_array(min_size=1, max_size=1))
@example(
    from_shapely([
        sg.Point([0.25, 0.25]),  # on line
        sg.Point([1, 1]),  # on vertex
        sg.Point([1.01, 1.01])  # on ray, just past vertex
    ]),
    from_shapely([sg.LineString([(0, 0), (1, 1), (2, 0)])]),
)
@hyp_settings
def test_points_intersects_line(gp_points, gp_line):
    # Get scalar Line
    sg_line = gp_line[0]

    # Compute expected intersection
    expected = gp_points.intersects(sg_line)

    # Create spatialpandas objects
    line = Line.from_shapely(sg_line)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(line) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(line)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(line, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(line),
        pd.Series(expected, index=points_series.index)
    )


@given(st_point_array(), st_line_array(min_size=1, max_size=1))
@example(
    from_shapely([
        sg.Point([0.25, 0.25]),  # on line
        sg.Point([1, 1]),  # on vertex
        sg.Point([1.01, 1.01])  # on ray, just past vertex
    ]),
    from_shapely([sg.MultiLineString([
        [(1, 0.5), (2, 0)],
        [(0, 0), (1, 1)],
    ])])
)
@hyp_settings
def test_points_intersects_multiline(gp_points, gp_multiline):
    # Get scalar MultiLine
    sg_multiline = gp_multiline[0]

    # Compute expected intersection
    expected = gp_points.intersects(sg_multiline)

    # Create spatialpandas objects
    multiline = MultiLine.from_shapely(sg_multiline)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(multiline) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(multiline)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(multiline, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(multiline),
        pd.Series(expected, index=points_series.index)
    )


@given(st_point_array(), st_polygon_array(min_size=1, max_size=1))
@hyp_settings
def test_points_intersects_polygon(gp_points, gp_polygon):
    # Get scalar Polygon
    sg_polygon = gp_polygon[0]

    # Compute expected intersection
    expected = gp_points.intersects(sg_polygon)

    # Create spatialpandas objects
    polygon = Polygon.from_shapely(sg_polygon)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(polygon) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(polygon)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(polygon, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(polygon),
        pd.Series(expected, index=points_series.index)
    )


@given(st_point_array(), st_multipolygon_array(min_size=1, max_size=1))
@hyp_settings
def test_points_intersects_multipolygon(gp_points, gp_multipolygon):
    # Get scalar MultiPolygon
    sg_multipolygon = gp_multipolygon[0]

    # Compute expected intersection
    expected = gp_points.intersects(sg_multipolygon)

    # Create spatialpandas objects
    multipolygon = MultiPolygon.from_shapely(sg_multipolygon)
    points = PointArray.from_geopandas(gp_points)
    points_series = GeoSeries(points, index=np.arange(10, 10 + len(points)))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(multipolygon) for point_el in points
    ])
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersect
    result = points.intersects(multipolygon)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(multipolygon, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test GeoSeries.intersects
    pd.testing.assert_series_equal(
        points_series.intersects(multipolygon),
        pd.Series(expected, index=points_series.index)
    )
