import numpy as np

from ..utils import ngjit

"""
Initially based on https://github.com/galtay/hilbert_curve, but specialized
for 2 dimensions with numba acceleration
"""


@ngjit
def _int_2_binary(v, width):
    """Return a binary byte array representation of `v` zero padded to `width`
    bits."""
    res = np.zeros(width, dtype=np.uint8)
    for i in range(width):
        res[width - i - 1] = v % 2
        v = v >> 1
    return res


@ngjit
def _binary_2_int(bin_vec):
    """Convert a binary byte array to an integer"""
    res = 0
    next_val = 1
    width = len(bin_vec)
    for i in range(width):
        res += next_val*bin_vec[width - i - 1]
        next_val <<= 1
    return res


@ngjit
def _hilbert_integer_to_transpose(p, h, n):
    """Store a hilbert integer (`h`) as its transpose (`x`).

    Args:
        p (int): iterations to use in the hilbert curve
        h (int): integer distance along hilbert curve
        n (int): number of dimensions
    Returns:
        x (list): transpose of h
                  (n components with values between 0 and 2**p-1)
    """
    h_bits = _int_2_binary(h, p * n)

    x = [_binary_2_int(h_bits[i::n]) for i in range(n)]
    return x


@ngjit
def _transpose_to_hilbert_integer(p, coord):
    """Restore a hilbert integer (`h`) from its transpose (`x`).

    Args:
        p (int): iterations to use in the hilbert curve
        x (list): transpose of h
                  (n components with values between 0 and 2**p-1)

    Returns:
        h (int): integer distance along hilbert curve
    """
    n = len(coord)
    bins = [_int_2_binary(v, p) for v in coord]
    concat = np.zeros(n*p, dtype=np.uint8)
    for i in range(p):
        for j in range(n):
            concat[n*i + j] = bins[j][i]

    h = _binary_2_int(concat)
    return h


@ngjit
def coordinate_from_distance(p, n, h):
    """Return the coordinate for a hilbert distance.

    Args:
        p (int): iterations to use in the hilbert curve
        n (int): number of dimensions
        h (int): integer distance along hilbert curve
    Returns:
        coord (list): Coordinate as length-n list
    """
    coord = _hilbert_integer_to_transpose(p, h, n)
    Z = 2 << (p-1)

    # Gray decode by H ^ (H/2)
    t = coord[n-1] >> 1
    for i in range(n-1, 0, -1):
        coord[i] ^= coord[i-1]
    coord[0] ^= t

    # Undo excess work
    Q = 2
    while Q != Z:
        P = Q - 1
        for i in range(n-1, -1, -1):
            if coord[i] & Q:
                # invert
                coord[0] ^= P
            else:
                # exchange
                t = (coord[0] ^ coord[i]) & P
                coord[0] ^= t
                coord[i] ^= t
        Q <<= 1

    return coord


@ngjit
def coordinates_from_distances(p, n, h):
    """Return the coordinates for an array of hilbert distances.

    Args:
        p (int): iterations to use in the hilbert curve
        n (int): number of dimensions
        h (ndarray): 1d array of integer distances along hilbert curve
    Returns:
        coords (list): 2d array of coordinate, each row a coordinate corresponding to
            associated distance value in input.
    """
    result = np.zeros((len(h), n), dtype=np.int64)

    for i in range(len(h)):
        result[i, :] = coordinate_from_distance(p, n, h[i])

    return result


@ngjit
def distance_from_coordinate(p, coord):
    """Return the hilbert distance for a given coordinate.

    Args:
        p (int): iterations to use in the hilbert curve
        coords (ndarray): coordinate as 1d array
    Returns:
        h (int): distance
    """
    n = len(coord)
    M = 1 << (p - 1)
    # Inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if coord[i] & Q:
                coord[0] ^= P
            else:
                t = (coord[0] ^ coord[i]) & P
                coord[0] ^= t
                coord[i] ^= t
        Q >>= 1
    # Gray encode
    for i in range(1, n):
        coord[i] ^= coord[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if coord[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coord[i] ^= t
    h = _transpose_to_hilbert_integer(p, coord)
    return h


@ngjit
def distances_from_coordinates(p, coords):
    """Return the hilbert distances for a given set of coordinates.

    Args:
        p (int): iterations to use in the hilbert curve
        coords (ndarray): 2d array of coordinates, one coordinate per row
    Returns:
        h (ndarray): 1d array of distances
    """
    coords = np.atleast_2d(coords).copy()
    result = np.zeros(coords.shape[0], dtype=np.int64)
    for i in range(coords.shape[0]):
        coord = coords[i, :]
        result[i] = distance_from_coordinate(p, coord)
    return result
