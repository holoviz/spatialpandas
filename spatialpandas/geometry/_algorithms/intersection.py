import numpy as np
from numba import prange

from ...geometry._algorithms.bounds import total_bounds_interleaved
from ...geometry._algorithms.orientation import triangle_orientation
from ...utils import ngjit, ngpjit


@ngjit
def segment_intersects_point(ax0, ay0, ax1, ay1, bx, by):
    """
    Test whether a 2-dimensional line segment intersects with a point

    Args:
        ax0, ay0: coordinates of start of segment
        ax1, ay1: coordinates of end of segment
        bx, by: coordinates of point

    Returns:
        True if segment intersects point, False otherwise
    """
    # Check bounds
    if bx < min(ax0, ax1) or bx > max(ax0, ax1):
        return False
    if by < min(ay0, ay1) or by > max(ay0, ay1):
        return False

    # Use cross product to test whether point is exactly on line
    # S is vector from segment start to segment end
    sx = ax1 - ax0
    sy = ay1 - ay0

    # P is vector from segment start to point
    px = bx - ax0
    py = by - ay0

    # Compute cross produce of S and P
    sxp = sx * py - sy * px

    return sxp == 0


@ngjit
def segments_intersect_1d(ax0, ax1, bx0, bx1):
    """
    Test whether two 1-dimensional line segments overlap
    Args:
        ax0, ax1: coords of endpoints of first segment
        bx0, bx1: coords of endpoints of second segment

    Returns:
        True if segments overlap, False otherwise
    """
    # swap inputs so that *x1 >= *x0
    if ax1 < ax0:
        ax0, ax1 = ax1, ax0
    if bx1 < bx0:
        bx0, bx1 = bx1, bx0

    return max(ax0, bx0) <= min(ax1, bx1)


@ngjit
def segments_intersect(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
    """
    Test whether two 2-dimensional line segments intersect

    Args:
        ax0, ay0: coordinates of start of first segment
        ax1, ay1: coordinates of end of first segment
        bx0, by0: coordinates of start of second segment
        bx1, by1: coordinates of end of second segment

    Returns:
        True if segments intersect, False otherwise
    """
    if not segments_intersect_1d(ax0, ax1, bx0, bx1):
        # x projection of segments do not intersect, segments do not cross
        return False
    if not segments_intersect_1d(ay0, ay1, by0, by1):
        # y projection of segments do not intersect, segments do not cross
        return False

    a_zero = ax0 == ax1 and ay0 == ay1
    b_zero = bx0 == bx1 and by0 == by1
    if a_zero and not b_zero and (
            ax0 == bx0 and ay0 == by0 or ax0 == bx1 and ay0 == by1
    ):
        # a is zero length line that is identical to an end point of b
        return True
    elif b_zero and not a_zero and (
            bx0 == ax0 and by0 == ay0 or bx0 == ax1 and by0 == ay1
    ):
        # a is zero length line that is identical to an end point of b
        return True
    elif a_zero or b_zero:
        # a or b is zero length and does not match a vertex of the other line
        return False

    b0_orientation = triangle_orientation(ax0, ay0, ax1, ay1, bx0, by0)
    b1_orientation = triangle_orientation(ax0, ay0, ax1, ay1, bx1, by1)
    if b0_orientation == 0 and b1_orientation == 0:
        # b0 and b1 lie on line from a0 to a1, segments are collinear and intersect
        return True
    elif b0_orientation == b1_orientation:
        # b0 and b1 lie on the same side of line from a0 to a1, segments do not
        # intersect
        return False

    a0_orientation = triangle_orientation(bx0, by0, bx1, by1, ax0, ay0)
    a1_orientation = triangle_orientation(bx0, by0, bx1, by1, ax1, ay1)
    if a0_orientation == 0 and a1_orientation == 0:
        # a0 and a1 lie on line from b0 to b1, segments are collinear and cross
        return True
    elif a0_orientation == a1_orientation:
        # a0 and a1 are on the same side of line from b0 to b1, segments do not cross
        return False

    return True


@ngjit
def point_intersects_polygon(x, y, values, value_offsets):
    """
    Test whether a point intersects with a polygon

    Args:
        x, y: coordinates of test point
        values: array of interleaved coordinates of the polygon and holes
        value_offsets: array of offsets into values that separate the rings that
            compose the polygon. The first ring is the outer shell and subsequent rings
            are holes contained in this shell.

    Returns:
        True if the test point intersects with the polygon, False otherwise
    """
    winding_number = 0
    for i in range(len(value_offsets) - 1):
        start = value_offsets[i]
        stop = value_offsets[i + 1]

        for k in range(start, stop - 2, 2):
            x0 = values[k]
            y0 = values[k + 1]
            x1 = values[k + 2]
            y1 = values[k + 3]

            if y1 == y0:
                # skip horizontal edges
                continue

            # Make sure the y1 > y0 and keep track of whether edge was
            # originally ascending vertically
            if y1 < y0:
                ascending = -1
                y0, y1 = y1, y0
                x0, x1 = x1, x0
            else:
                ascending = 1

            # Reject edges that are fully above, below, or to the left of test point
            if y0 >= y or y1 < y or (x0 < x and x1 < x):
                continue

            if x0 >= x and x1 >= x:
                # Edge is fully to the right of test point, so we know that a ray to
                # the right will intersect the edge
                winding_number += ascending
            else:
                # Check if edge is to the right of test point using cross product
                # A is vector from test point to lower vertex
                ax = x0 - x
                ay = y0 - y

                # B is vector from test point to upper vertex
                bx = x1 - x
                by = y1 - y

                # Compute cross produce of A and B
                axb = ax * by - ay * bx

                if axb > 0 or (axb == 0 and ascending):
                    # Edge intersects with ray
                    winding_number += ascending

    return winding_number != 0


@ngpjit
def multipoints_intersect_bounds(
        x0, y0, x1, y1, flat_values, start_offsets, stop_offsets, result
):
    """
    Test whether each multipoint in a collection of multipoints intersects with
    the supplied bounds

    Args:
        x0, y0, x1, y1: Bounds coordinates
        flat_values: Interleaved point coordinates
        start_offsets, stop_offsets:  start and stop offsets into flat_values
            separating individual multipoints
        result: boolean array to be provided by the caller into which intersection
            results will be written; must be at least as long as start_offsets

    Returns:
        None
    """
    # Initialize results
    n = len(start_offsets)
    result.fill(False)

    # Orient rectangle
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1, = y1, y0

    for i in prange(n):
        start = start_offsets[i]
        stop = stop_offsets[i]

        # Check for points in rect
        point_in_rect = False
        for j in range(start, stop, 2):
            x = flat_values[j]
            y = flat_values[j + 1]
            if x0 <= x and x <= x1 and y0 <= y and y <= y1:
                point_in_rect = True
                break

        if point_in_rect:
            result[i] = True
            continue

    return result


@ngjit
def _perform_line_intersect_bounds(
        i, x0, y0, x1, y1, flat_values, start_offsets, stop_offsets, result
):
    start = start_offsets[i]
    stop = stop_offsets[i]

    # compute bounding box for line
    bounds = total_bounds_interleaved(flat_values[start:stop])

    if bounds[0] > x1 or bounds[1] > y1 or bounds[2] < x0 or bounds[3] < y0:
        # bounds outside of rect, does not intersect
        return

    if (bounds[0] >= x0 and bounds[2] <= x1 or
            bounds[1] >= y0 and bounds[3] <= y1):
        # bounds is fully contained in rect when both are projected onto the
        # x or y axis
        result[i] = True
        return

    # Check for vertices in rect
    vert_in_rect = False
    for j in range(start, stop, 2):
        x = flat_values[j]
        y = flat_values[j + 1]
        if x0 <= x and x <= x1 and y0 <= y and y <= y1:
            vert_in_rect = True
            break

    if vert_in_rect:
        result[i] = True
        return

    # Check for segment that crosses rectangle edge
    segment_intersects = False
    for j in range(start, stop - 2, 2):
        ex0 = flat_values[j]
        ey0 = flat_values[j + 1]
        ex1 = flat_values[j + 2]
        ey1 = flat_values[j + 3]

        # top
        if segments_intersect(ex0, ey0, ex1, ey1, x0, y1, x1, y1):
            segment_intersects = True
            break

        # bottom
        if segments_intersect(ex0, ey0, ex1, ey1, x0, y0, x1, y0):
            segment_intersects = True
            break

        # left
        if segments_intersect(ex0, ey0, ex1, ey1, x0, y0, x0, y1):
            segment_intersects = True
            break

        # right
        if segments_intersect(ex0, ey0, ex1, ey1, x1, y0, x1, y1):
            segment_intersects = True
            break

    if segment_intersects:
        result[i] = True


@ngjit
def lines_intersect_bounds(
        x0, y0, x1, y1, flat_values, start_offsets, stop_offsets, result
):
    """
    Test whether each line in a collection of lines intersects with the supplied bounds

    Args:
        x0, y0, x1, y1: Bounds coordinates
        flat_values: Interleaved line coordinates
        start_offsets, stop_offsets:  start and stop offsets into flat_values
            separating individual lines
        result: boolean array to be provided by the caller into which intersection
            results will be written; must be at least as long as start_offsets

    Returns:
        None
    """
    # Initialize results
    n = len(start_offsets)
    result.fill(False)

    # Orient rectangle
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1, = y1, y0

    if x0 == x1 or y0 == y1:
        # Zero width/height rect does not intersect with anything
        return

    for i in range(n):
        _perform_line_intersect_bounds(
            i, x0, y0, x1, y1, flat_values, start_offsets, stop_offsets, result
        )

    return result


@ngjit
def multilines_intersect_bounds(
        x0, y0, x1, y1, flat_values, start_offsets0, stop_offsets0, offsets1, result
):
    """
    Test whether each multiline in a collection of multilines intersects with the
    supplied bounds

    Args:
        x0, y0, x1, y1: Bounds coordinates
        flat_values: Interleaved line coordinates
        start_offsets0, stop_offsets0:  start and stop offsets into offsets1
            separating individual multilines
        offsets1: Offsets into flat_values separating individual lines
        result: boolean array to be provided by the caller into which intersection
            results will be written; must be at least as long as start_offsets

    Returns:
        None
    """
    # Initialize results
    n = len(start_offsets0)
    result.fill(False)

    if len(start_offsets0) < 1:
        # Empty array
        return

    # Orient rectangle
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1, = y1, y0

    if x0 == x1 or y0 == y1:
        # Zero width/height rect does not intersect with anything
        return

    # Populate results
    for i in range(n):
        # Numba has issues with following line when jit(parallel=True) is used:
        # Invalid use of Function(<intrinsic wrap_index>) with argument(s) of type(s):
        #   (uint32, int64)
        element_offsets = offsets1[start_offsets0[i]:stop_offsets0[i] + 1]
        num_lines = len(element_offsets) - 1
        element_result = np.zeros(num_lines, dtype=np.bool_)
        for j in range(num_lines):
            _perform_line_intersect_bounds(
                j, x0, y0, x1, y1,
                flat_values, element_offsets[:-1], element_offsets[1:], element_result
            )
        result[i] = element_result.any()


@ngjit
def _perform_polygon_intersect_bounds(i, x0, y0, x1, y1, flat_values, start_offsets0,
                                      stop_offsets0, offsets1, result):
    start0 = start_offsets0[i]
    stop0 = stop_offsets0[i]
    start1 = offsets1[start0]
    stop1 = offsets1[stop0]

    # compute bounding box for polygon.
    bounds = total_bounds_interleaved(flat_values[start1:stop1])

    if bounds[0] > x1 or bounds[1] > y1 or bounds[2] < x0 or bounds[3] < y0:
        # bounds outside of rect, does not intersect
        return

    if (bounds[0] >= x0 and bounds[2] <= x1 or
            bounds[1] >= y0 and bounds[3] <= y1):
        # bounds is fully contained in rect when both are projected onto the
        # x or y axis
        result[i] = True
        return

        # Check for vertices in rect
    vert_in_rect = False
    for k in range(start1, stop1, 2):
        x = flat_values[k]
        y = flat_values[k + 1]
        if x0 <= x and x <= x1 and y0 <= y and y <= y1:
            vert_in_rect = True
            break

    if vert_in_rect:
        result[i] = True
        return

    # Check for segment that crosses rectangle edge
    segment_intersects = False
    for j in range(start0, stop0):
        for k in range(offsets1[j], offsets1[j + 1] - 2, 2):
            ex0 = flat_values[k]
            ey0 = flat_values[k + 1]
            ex1 = flat_values[k + 2]
            ey1 = flat_values[k + 3]

            # top
            if segments_intersect(ex0, ey0, ex1, ey1, x0, y1, x1, y1):
                segment_intersects = True
                break

            # bottom
            if segments_intersect(ex0, ey0, ex1, ey1, x0, y0, x1, y0):
                segment_intersects = True
                break

            # left
            if segments_intersect(ex0, ey0, ex1, ey1, x0, y0, x0, y1):
                segment_intersects = True
                break

            # right
            if segments_intersect(ex0, ey0, ex1, ey1, x1, y0, x1, y1):
                segment_intersects = True
                break

        if segment_intersects:
            result[i] = True

    if segment_intersects:
        return

        # Check if a rectangle corners is in rect
    polygon_offsets = offsets1[start0:stop0 + 1]
    if point_intersects_polygon(x0, y0, flat_values, polygon_offsets):
        result[i] = True
        return
    if point_intersects_polygon(x1, y0, flat_values, polygon_offsets):
        result[i] = True
        return
    if point_intersects_polygon(x1, y1, flat_values, polygon_offsets):
        result[i] = True
        return
    if point_intersects_polygon(x0, y1, flat_values, polygon_offsets):
        result[i] = True
        return


@ngjit
def polygons_intersect_bounds(
        x0, y0, x1, y1, flat_values, start_offsets0, stop_offsets0, offsets1, result
):
    """
    Test whether each polygon in a collection of polygons intersects with the
    supplied bounds

    Args:
        x0, y0, x1, y1: Bounds coordinates
        flat_values: Interleaved vertex coordinates
        start_offsets0, stop_offsets0: start and stop offsets into offsets1 separating
            individual polygons
        offsets1:  offsets into flat_values separating individual polygons rings
        result: boolean array to be provided by the caller into which intersection
            results will be written; must be at least as long as start_offsets

    Returns:
        None
    """
    # Initialize results
    n = len(start_offsets0)
    result.fill(False)

    # Orient rectangle
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1, = y1, y0

    if x0 == x1 or y0 == y1:
        # Zero width/height rect does not intersect with anything
        return

    for i in range(n):
        _perform_polygon_intersect_bounds(
            i, x0, y0, x1, y1, flat_values,
            start_offsets0, stop_offsets0, offsets1, result
        )

    return result


@ngjit
def multipolygons_intersect_bounds(
        x0, y0, x1, y1, flat_values,
        start_offsets0, stop_offsets0, offsets1, offsets2, result
):
    """
    Test whether each multipolygon in a collection of multipolygons intersects with
    the supplied bounds

    Args:
        x0, y0, x1, y1: Bounds coordinates
        flat_values: Interleaved vertex coordinates
        start_offsets0, stop_offsets0: start and stop offsets into offsets1 separating
            individual multipolygons
        offsets1: offsets into offsets2 separating individual polygons
        offsets2: offsets into flat_values separating individual polygon rings
        result: boolean array to be provided by the caller into which intersection
            results will be written; must be at least as long as start_offsets

    Returns:
        None
    """

    # Initialize results
    n = len(start_offsets0)
    result.fill(False)

    if len(start_offsets0) < 1:
        # Empty array
        return

    # Orient rectangle
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1, = y1, y0

    if x0 == x1 or y0 == y1:
        # Zero width/height rect does not intersect with anything
        return

    # Populate results
    for i in range(n):
        polygon_offsets = offsets1[start_offsets0[i]:stop_offsets0[i] + 1]
        num_polys = len(polygon_offsets) - 1
        element_result = np.zeros(num_polys, dtype=np.bool_)
        for j in range(num_polys):
            _perform_polygon_intersect_bounds(
                j, x0, y0, x1, y1, flat_values,
                polygon_offsets[:-1], polygon_offsets[1:], offsets2, element_result
            )

        result[i] = element_result.any()
