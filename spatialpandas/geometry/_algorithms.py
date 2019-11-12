from math import sqrt

import numpy as np
from numba import jit, prange

from spatialpandas.utils import ngjit


@ngjit
def bounds_interleaved(values):
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

    return (xmin, ymin, xmax, ymax)


@ngjit
def bounds_interleaved_1d(values, offset):
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

    return (vmin, vmax)


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


@jit(nogil=True, nopython=True, parallel=True)
def geometry_map_nested1(
        fn, result, result_offset, values, value_offsets, missing
):
    assert len(value_offsets) == 1
    value_offsets0 = value_offsets[0]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i + result_offset] = fn(values, value_offsets0[i:i + 2])


@jit(nogil=True, nopython=True, parallel=True)
def geometry_map_nested2(
        fn, result, result_offset, values, value_offsets, missing
):
    assert len(value_offsets) == 2
    value_offsets0 = value_offsets[0]
    value_offsets1 = value_offsets[1]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = value_offsets0[i]
            stop = value_offsets0[i + 1]
            result[i + result_offset] = fn(values, value_offsets1[start:stop + 1])


@jit(nogil=True, nopython=True, parallel=True)
def geometry_map_nested3(
        fn, result, result_offset, values, value_offsets, missing
):
    assert len(value_offsets) == 3
    value_offsets0 = value_offsets[0]
    value_offsets1 = value_offsets[1]
    value_offsets2 = value_offsets[2]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = value_offsets1[value_offsets0[i]]
            stop = value_offsets1[value_offsets0[i + 1]]
            result[i + result_offset] = fn(values, value_offsets2[start:stop + 1])


@jit(nopython=True, nogil=True)
def _lexographic_lt0(a1, a2):
    """
    Compare two 1D numpy arrays lexographically
    Parameters
    ----------
    a1: ndarray
        1D numpy array
    a2: ndarray
        1D numpy array

    Returns
    -------
    comparison:
        True if a1 < a2, False otherwise
    """
    for e1, e2 in zip(a1, a2):
        if e1 < e2:
            return True
        elif e1 > e2:
            return False
    return len(a1) < len(a2)


def _lexographic_lt(a1, a2):
    if a1.dtype != np.object and a1.dtype != np.object:
        # a1 and a2 primitive
        return _lexographic_lt0(a1, a2)
    elif a1.dtype == np.object and a1.dtype == np.object:
        # a1 and a2 object, process recursively
        for e1, e2 in zip(a1, a2):
            if _lexographic_lt(e1, e2):
                return True
            elif _lexographic_lt(e2, e1):
                return False
        return len(a1) < len(a2)
    elif a1.dtype != np.object:
        # a2 is object array, a1 primitive
        return True
    else:
        # a1 is object array, a2 primitive
        return False


@ngjit
def _extract_isnull_bytemap(bitmap, bitmap_length, bitmap_offset, dst_offset, dst):
    """
    Note: Copied from fletcher: See NOTICE for license info

    (internal) write the values of a valid bitmap as bytes to a pre-allocatored
    isnull bytemap.

    Parameters
    ----------
    bitmap: pyarrow.Buffer
        bitmap where a set bit indicates that a value is valid
    bitmap_length: int
        Number of bits to read from the bitmap
    bitmap_offset: int
        Number of bits to skip from the beginning of the bitmap.
    dst_offset: int
        Number of bytes to skip from the beginning of the output
    dst: numpy.array(dtype=bool)
        Pre-allocated numpy array where a byte is set when a value is null
    """
    for i in range(bitmap_length):
        idx = bitmap_offset + i
        byte_idx = idx // 8
        bit_mask = 1 << (idx % 8)
        dst[dst_offset + i] = (bitmap[byte_idx] & bit_mask) == 0


def extract_isnull_bytemap(list_array):
    """
    Note: Copied from fletcher: See NOTICE for license info

    Extract the valid bitmaps of a chunked array into numpy isnull bytemaps.

    Parameters
    ----------
    chunked_array: pyarrow.ChunkedArray

    Returns
    -------
    valid_bytemap: numpy.array
    """
    result = np.zeros(len(list_array), dtype=bool)

    offset = 0
    chunk = list_array
    valid_bitmap = chunk.buffers()[0]
    if valid_bitmap:
        buf = memoryview(valid_bitmap)
        _extract_isnull_bytemap(buf, len(chunk), chunk.offset, offset, result)
    else:
        return np.full(len(list_array), False)

    return result
