import numpy as np
import pandas as pd
from hypothesis import given
from spatialpandas.geometry import PointArray, Point
from tests.geometry.strategies import st_point_array, st_bounds, hyp_settings


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

    # Test PointArray.intersect
    result = points.intersects(point)
    np.testing.assert_equal(result, expected)

    # Test PointArray.intersects with inds
    inds = np.flipud(np.arange(0, len(points)))
    result = points.intersects(point, inds)
    np.testing.assert_equal(result, np.flipud(expected))

    # Test Point.intersects
    result = np.array([
        point_el.intersects(point) for point_el in points
    ])
    np.testing.assert_equal(result, expected)
