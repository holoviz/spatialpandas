import numpy as np
import pytest
from hypothesis import example, given
from shapely import geometry as sg

from ..strategies import (
    coord,
    hyp_settings,
    st_bounds,
    st_line_array,
    st_multiline_array,
    st_multipoint_array,
    st_multipolygon_array,
    st_point_array,
    st_points,
    st_polygon,
    st_polygon_array,
)
from spatialpandas.geometry import (
    LineArray,
    MultiLineArray,
    MultiPointArray,
    MultiPolygonArray,
    PointArray,
    Polygon,
    PolygonArray,
)
from spatialpandas.geometry._algorithms.intersection import (
    point_intersects_polygon,
    segment_intersects_point,
    segments_intersect,
)


@given(coord, coord, coord, coord, coord, coord)
@example(0, 0, 1, 1, 0.25, 0.25)  # Point on line
@example(0, 0, 1, 1, 0.2501, 0.25)  # Point just off of line
@example(0, 0, 1, 1, 1, 1)  # Point on end of segment
@example(0, 0, 1, 1, 0, 0)  # Point on start of segment
@example(0, 0, 1, 1, 1.001, 1.001)  # Point on ray, just past end of segment
@example(0, 0, 1, 1, -0.001, -0.001)  # Point on ray, just before start of segment
@hyp_settings
def test_segment_intersects_point(ax0, ay0, ax1, ay1, bx, by):
    result1 = segment_intersects_point(ax0, ay0, ax1, ay1, bx, by)
    result2 = segment_intersects_point(ax1, ay1, ax0, ay0, bx, by)
    # Ensure we get the same answer with line order flipped
    assert result1 == result2

    # Use shapely polygon to compute expected intersection
    line = sg.LineString([(ax0, ay0), (ax1, ay1)])
    point = sg.Point([bx, by])
    expected = line.intersects(point)
    assert result1 == expected


@given(coord, coord, coord, coord, coord, coord, coord, coord)
@hyp_settings
def test_segment_intersection(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    result1 = segments_intersect(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1)
    result2 = segments_intersect(bx0, by0, bx1, by1, ax0, ay0, ax1, ay1)
    # Ensure we get the same answer with line order flipped
    assert result1 == result2

    # Use shapely polygon to compute expected intersection
    line1 = sg.LineString([(ax0, ay0), (ax1, ay1)])
    line2 = sg.LineString([(bx0, by0), (bx1, by1)])
    expected = line1.intersects(line2)
    assert result1 == expected


@given(st_polygon(), st_points)
@hyp_settings
def test_point_intersects_polygon(sg_polygon, points):
    polygon = Polygon.from_shapely(sg_polygon)

    for r in range(points.shape[0]):
        x, y = points[r, :]
        result = point_intersects_polygon(
            x, y, polygon.buffer_values, polygon.buffer_inner_offsets
        )

        expected = sg_polygon.intersects(sg.Point([x, y]))
        assert expected == result


@given(st_point_array(), st_bounds())
@hyp_settings
def test_point_intersects_rect(gp_point, rect):
    sg_rect = sg.box(*rect)
    expected = gp_point.intersects(sg_rect)
    points = PointArray.from_geopandas(gp_point)

    # Test MultiPointArray.intersects_rect
    result = points.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test MultiPointArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test MultiPoint.intersects_rect
    result = np.array([
        point.intersects_bounds(rect) for point in points
    ])
    np.testing.assert_equal(result, expected)


@given(st_multipoint_array(), st_bounds())
@hyp_settings
def test_multipoint_intersects_rect(gp_multipoint, rect):
    sg_rect = sg.box(*rect)
    expected = gp_multipoint.intersects(sg_rect)
    multipoints = MultiPointArray.from_geopandas(gp_multipoint)

    # Test MultiPointArray.intersects_rect
    result = multipoints.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test MultiPointArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(multipoints)))
    result = multipoints.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test MultiPoint.intersects_rect
    result = np.array([
        multipoint.intersects_bounds(rect) for multipoint in multipoints
    ])
    np.testing.assert_equal(result, expected)


@given(st_line_array(), st_bounds())
@hyp_settings
def test_line_intersects_rect(gp_line, rect):
    sg_rect = sg.box(*rect)

    expected = gp_line.intersects(sg_rect)
    lines = LineArray.from_geopandas(gp_line)

    # Test LineArray.intersects_rect
    result = lines.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test LineArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(lines)))
    result = lines.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test Line.intersects_rect
    result = np.array([line.intersects_bounds(rect) for line in lines])
    np.testing.assert_equal(result, expected)


@given(st_multiline_array(), st_bounds())
@hyp_settings
def test_multiline_intersects_rect(gp_multiline, rect):
    sg_rect = sg.box(*rect)

    expected = gp_multiline.intersects(sg_rect)
    multilines = MultiLineArray.from_geopandas(gp_multiline)

    # Test MultiLineArray.intersects_rect
    result = multilines.intersects_bounds(rect)
    if not np.array_equal(result, expected):
        multilines.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test MultiLineArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(multilines)))
    result = multilines.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test MultiLine.intersects_rect
    result = np.array([
        multiline.intersects_bounds(rect) for multiline in multilines
    ])
    np.testing.assert_equal(result, expected)


@pytest.mark.slow
@given(
    st_polygon_array(),
    st_bounds(
        x_min=-150, x_max=150, y_min=-150, y_max=150
    )
)
@hyp_settings
def test_polygon_intersects_rect(gp_polygon, rect):
    sg_rect = sg.box(*rect)

    expected = gp_polygon.intersects(sg_rect)
    polygons = PolygonArray.from_geopandas(gp_polygon)

    # Test PolygonArray.intersects_rect
    result = polygons.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test PolygonArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(polygons)))
    result = polygons.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test Polygon.intersects_rect
    result = np.array([
        polygon.intersects_bounds(rect) for polygon in polygons
    ])
    np.testing.assert_equal(result, expected)


@pytest.mark.slow
@given(
    st_multipolygon_array(),
    st_bounds(
        x_min=-150, x_max=150, y_min=-150, y_max=150
    )
)
@hyp_settings
def test_multipolygon_intersects_rect(gp_multipolygon, rect):
    sg_rect = sg.box(*rect)

    expected = gp_multipolygon.intersects(sg_rect)
    multipolygons = MultiPolygonArray.from_geopandas(gp_multipolygon)

    # Test MultiPolygonArray.intersects_rect
    result = multipolygons.intersects_bounds(rect)
    np.testing.assert_equal(result, expected)

    # Test MultiPolygonArray.intersects_rect with inds
    inds = np.flipud(np.arange(0, len(multipolygons)))
    result = multipolygons.intersects_bounds(rect, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test MultiPolygon.intersects_rect
    result = np.array([
        multipolygon.intersects_bounds(rect) for multipolygon in multipolygons
    ])
    np.testing.assert_equal(result, expected)
