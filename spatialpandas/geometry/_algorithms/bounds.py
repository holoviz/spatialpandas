import numpy as np

from ...utils import ngjit


@ngjit
def total_bounds_interleaved(values):
    """
    compute bounds
    """
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf

    for i in range(0, len(values), 2):
        x = values[i]
        if np.isfinite(x):
            xmin = min(xmin, x)
            xmax = max(xmax, x)

        y = values[i + 1]
        if np.isfinite(y):
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    # replace infinite values with nan in the case no finite values were found
    if not np.isfinite(xmin):
        xmin = xmax = np.nan
    if not np.isfinite(ymin):
        ymin = ymax = np.nan

    return (xmin, ymin, xmax, ymax)


@ngjit
def total_bounds_interleaved_1d(values, offset):
    """
    compute bounds
    """
    vmin = np.inf
    vmax = -np.inf

    for i in range(0, len(values), 2):
        v = values[i + offset]
        if np.isfinite(v):
            vmin = min(vmin, v)
            vmax = max(vmax, v)

    if np.isfinite(vmin):
        return (vmin, vmax)
    else:
        return (np.nan, np.nan)


@ngjit
def bounds_interleaved(flat_values, flat_value_offsets):
    # Initialize bounds array that will be populated and returned
    bounds = np.full((len(flat_value_offsets) - 1, 4), np.nan, dtype=np.float64)

    for i in range(len(flat_value_offsets) - 1):
        start = flat_value_offsets[i]
        stop = flat_value_offsets[i + 1]
        bounds[i, :] = total_bounds_interleaved(flat_values[start:stop])

    return bounds
