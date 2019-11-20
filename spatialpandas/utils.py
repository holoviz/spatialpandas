import numpy as np
from numba import jit

ngjit = jit(nopython=True, nogil=True)
ngpjit = jit(nopython=True, nogil=True, parallel=True)


@ngjit
def _data2coord(vals, val_range, n):
    """
    Convert an array of values from continuous data coordinates to discrete
    integer coordinates

    Args:
        vals: Array of continuous data coordinate to be converted to discrete
            integer coordinates
        val_range: Tuple of start (val_range[0]) and stop (val_range[1]) range in
            continuous data coordinates
        n: The integer number of discrete distance coordinates

    Returns:
        unsigned integer array of discrete coordinates
    """
    x_width = val_range[1] - val_range[0]
    res = ((vals - val_range[0]) * (n / x_width)).astype(np.int64)

    # clip
    res[res < 0] = 0
    res[res > n - 1] = n - 1
    return res
