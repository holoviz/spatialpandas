from hypothesis import given
from shapely import geometry as sg

from ..strategies import coord, hyp_settings
from spatialpandas.geometry._algorithms.orientation import triangle_orientation


@given(coord, coord, coord, coord, coord, coord)
@hyp_settings
def test_triangle_orientation(ax, ay, bx, by, cx, cy):
    result = triangle_orientation(ax, ay, bx, by, cx, cy)

    # Use shapely polygon to compute expected orientation
    sg_poly = sg.Polygon([(ax, ay), (bx, by), (cx, cy), (ax, ay)])

    if sg_poly.area == 0:
        expected = 0
    else:
        expected = 1 if sg_poly.exterior.is_ccw else -1

    assert result == expected
