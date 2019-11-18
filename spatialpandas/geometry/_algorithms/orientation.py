from spatialpandas.utils import ngjit


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
