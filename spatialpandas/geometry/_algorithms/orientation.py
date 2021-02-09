import numpy as np

from ...geometry._algorithms.measures import compute_area
from ...utils import ngjit


@ngjit
def triangle_orientation(ax, ay, bx, by, cx, cy):
    """
    Calculate orientation of triangle
    Args:
        ax, ay: coords of first point
        bx, by: coords of second point
        cx, cy: coords of third point

    Returns:
        +1 if counter clockwise
         0 if colinear
        -1 if clockwise
    """
    ab_x, ab_y = bx - ax, by - ay
    ac_x, ac_y = cx - ax, cy - ay

    # compute cross product: ab x bc
    ab_x_ac = (ab_x * ac_y) - (ab_y * ac_x)

    if ab_x_ac > 0:
        # Counter clockwise
        return 1
    elif ab_x_ac < 0:
        # Clockwise
        return -1
    else:
        # Collinear
        return 0


@ngjit
def orient_polygons(values, polygon_offsets, ring_offsets):
    """
    Orient polygons so that exterior is in CCW order and interior rings (holes) are in
    CW order.

    This function mutates the values array

    Args:
        values: Ring coordinates
        polygon_offsets: Offsets into ring_offsets of first ring in each polygon
        ring_offsets: Offsets into values of the start of each ring
    """
    num_rings = len(ring_offsets) - 1

    # Compute expected orientation of rings
    expected_ccw = np.zeros(len(ring_offsets) - 1, dtype=np.bool_)
    expected_ccw[polygon_offsets[:-1]] = True

    # Compute actual orientation of rings
    is_ccw = np.zeros(num_rings)
    for i in range(num_rings):
        is_ccw[i] = compute_area(values, ring_offsets[i:i + 2]) >= 0

    # Compute indices of rings to flip
    flip_inds = np.nonzero(is_ccw != expected_ccw)
    ring_starts = ring_offsets[:-1]
    ring_stops = ring_offsets[1:]
    flip_starts = ring_starts[flip_inds]
    flip_stops = ring_stops[flip_inds]

    for i in range(len(flip_starts)):
        flip_start = flip_starts[i]
        flip_stop = flip_stops[i]

        xs = values[flip_start:flip_stop:2]
        ys = values[flip_start + 1:flip_stop:2]
        values[flip_start:flip_stop:2] = xs[::-1]
        values[flip_start + 1:flip_stop:2] = ys[::-1]
