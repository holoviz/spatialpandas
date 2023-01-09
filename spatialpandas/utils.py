import warnings

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


def _asarray_maybe_ragged(input):
    """Convert input into a single numpy array, even if it is a ragged array.

    Prior to numpy 1.24 just np.asarray(input) suffices as it emits a
    np.VisibleDeprecationWarning if the input is ragged, i.e. can't be
    converted to a single numpy array, but still completes the conversion by
    creating a numpy array of multiple subarrays. From 1.24 onwards attempting
    to do this raises a ValueError instead. This function therefore tries the
    simple conversion and if this fails uses the ragged-supporting conversion
    of np.asarray(input, type=object).

    To demonstrate that this works with numpy < 1.24 it converts
    VisibleDeprecationWarnings into errors so that they are handled the same
    as for numpy >= 1.24.

    Args:
        input: ArrayLike | list[ArrayLike | None]

    Returns:
        NumPy array.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.VisibleDeprecationWarning)
        try:
            array = np.asarray(input)
        except (ValueError, np.VisibleDeprecationWarning):
            array = np.asarray(input, dtype=object)

    return array
