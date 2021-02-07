from math import sqrt

import numpy as np

from ...utils import ngjit


@ngjit
def compute_line_length(values, value_offsets):
    total_len = 0.0
    for offset_ind in range(len(value_offsets) - 1):
        start = value_offsets[offset_ind]
        stop = value_offsets[offset_ind + 1]
        x0 = values[start]
        y0 = values[start + 1]

        for i in range(start + 2, stop, 2):
            x1 = values[i]
            y1 = values[i + 1]

            if (np.isfinite(x0) and np.isfinite(y0) and
                    np.isfinite(x1) and np.isfinite(y1)):
                total_len += sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            x0 = x1
            y0 = y1

    return total_len


@ngjit
def compute_area(values, value_offsets):
    area = 0.0

    for offset_ind in range(len(value_offsets) - 1):
        start = value_offsets[offset_ind]
        stop = value_offsets[offset_ind + 1]
        poly_length = stop - start

        if poly_length < 6:
            # A degenerate polygon, zero area
            continue

        for k in range(start, stop - 4, 2):
            i, j = k + 2, k + 4
            ix = values[i]
            jy = values[j + 1]
            ky = values[k + 1]

            area += ix * (jy - ky)

        # wrap-around term for polygon
        firstx = values[start]
        secondy = values[start + 3]
        lasty = values[stop - 3]
        area += firstx * (secondy - lasty)

    return area / 2.0
