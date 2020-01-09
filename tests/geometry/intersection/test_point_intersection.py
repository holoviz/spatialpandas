import numpy as np
import pandas as pd
from geopandas.array import from_shapely
from hypothesis import given
from spatialpandas.geometry import PointArray, Point, MultiPoint
from tests.geometry.strategies import st_point_array, st_multipoint_array, hyp_settings


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
