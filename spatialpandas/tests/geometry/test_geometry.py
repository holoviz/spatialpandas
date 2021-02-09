import numpy as np

from spatialpandas.geometry import (
    Line,
    LineArray,
    MultiPoint,
    MultiPointArray,
    MultiPolygon,
    MultiPolygonArray,
    Point,
    PointArray,
    Polygon,
    PolygonArray,
    Ring,
)

unit_square_cw = np.array([1, 1,  1, 2,  2, 2,  2, 1,  1, 1], dtype='float64')
large_square_ccw = np.array([0, 0, 3, 0, 3, 3, 0, 3, 0, 0], dtype='float64')


def test_point():
    point = Point([1, 2])
    assert point.length == 0.0
    assert point.area == 0.0


def test_point_array():
    points = PointArray(unit_square_cw)
    np.testing.assert_equal(points.length, [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_equal(points.area, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert points.total_bounds == (1, 1, 2, 2)


def test_multipoint():
    points = MultiPoint(unit_square_cw)
    assert points.length == 0.0
    assert points.area == 0.0


def test_multipoint_array():
    points = MultiPointArray([
        unit_square_cw,
        large_square_ccw,
        np.concatenate([large_square_ccw, unit_square_cw])
    ])

    np.testing.assert_equal(points.length, [0.0, 0.0, 0.0])
    np.testing.assert_equal(points.area, [0.0, 0.0, 0.0])
    assert points.total_bounds == (0.0, 0.0, 3.0, 3.0)


def test_lines():
    lines = Line(unit_square_cw)
    assert lines.length == 4.0
    assert lines.area == 0.0


def test_lines_array():
    lines = LineArray([
        unit_square_cw,
        large_square_ccw,
        np.concatenate([large_square_ccw, [np.nan, np.nan], unit_square_cw])
    ])

    np.testing.assert_equal(lines.length, [4.0, 12.0, 16.0])
    np.testing.assert_equal(lines.area, [0.0, 0.0, 0.0])
    assert lines.total_bounds == (0.0, 0.0, 3.0, 3.0)


def test_ring():
    lines = Ring(unit_square_cw)
    assert lines.length == 4.0
    assert lines.area == 0.0


def test_ring_array():
    lines = LineArray([
        unit_square_cw,
        large_square_ccw,
    ])

    np.testing.assert_equal(lines.length, [4.0, 12.0])
    np.testing.assert_equal(lines.area, [0.0, 0.0])
    assert lines.total_bounds == (0.0, 0.0, 3.0, 3.0)


def test_polygon():
    polygon = Polygon([large_square_ccw, unit_square_cw])
    assert polygon.length == 16.0
    assert polygon.area == 8.0


def test_polygon_array():
    polygons = PolygonArray([
        [large_square_ccw],
        [large_square_ccw, unit_square_cw],
        [unit_square_cw]
    ])
    np.testing.assert_equal(polygons.length, [12.0, 16.0, 4.0])
    np.testing.assert_equal(polygons.area, [9.0, 8.0, -1.0])
    assert polygons.total_bounds == (0.0, 0.0, 3.0, 3.0)


def test_multipolygon():
    multipolygon = MultiPolygon([
        [large_square_ccw, unit_square_cw],
        [large_square_ccw + 4.0]
    ])
    assert multipolygon.length == 28.0
    assert multipolygon.area == 17.0


def test_multipolygon_array():
    multipolygon = MultiPolygonArray([
        [
            [large_square_ccw, unit_square_cw],
            [large_square_ccw + 4.0]
        ], [
            [large_square_ccw + 8.0]
        ]
    ])
    np.testing.assert_equal(multipolygon.length, [28.0, 12.0])
    np.testing.assert_equal(multipolygon.area, [17.0, 9.0])
    assert multipolygon.total_bounds == (0.0, 0.0, 11.0, 11.0)
