from itertools import product

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except ImportError:
    HilbertCurve = None

# ### hypothesis settings ###
from spatialpandas.spatialindex.hilbert_curve import (
    coordinate_from_distance,
    coordinates_from_distances,
    distance_from_coordinate,
    distances_from_coordinates,
)

hyp_settings = settings(deadline=None)

# ### strategies ###
st_p = st.integers(min_value=1, max_value=5)
st_n = st.integers(min_value=1, max_value=3)

# ### Hypothesis tests ###
@given(st_p, st_n)
@hyp_settings
@pytest.mark.skipif(HilbertCurve is None, reason="hilbertcurve not installed")
def test_coordinates_from_distance(p, n):
    # Build vector of possible distances
    distances = np.arange(2 ** (n * p), dtype=np.int64)

    # Compute coordinates as vector result
    vector_result = coordinates_from_distances(p, n, distances)

    # Compare with reference
    reference_hc = HilbertCurve(p, n)
    for i, distance in enumerate(distances):
        # Reference
        expected = tuple(reference_hc.point_from_distance(distance))

        # Scalar result
        scalar_result = tuple(coordinate_from_distance(p, n, distance))
        assert scalar_result == expected

        # Vector result
        assert tuple(vector_result[i, :]) == expected


@given(st_p, st_n)
@hyp_settings
@pytest.mark.skipif(HilbertCurve is None, reason="hilbertcurve not installed")
def test_distance_from_coordinates(p, n):
    side_len = 2 ** p

    # build matrix of all possible coordinate
    coords = np.array(list(product(range(side_len), repeat=n)))

    # Compute distances as vector result
    vector_result = distances_from_coordinates(p, coords)

    # Compare with reference
    reference_hc = HilbertCurve(p, n)
    for i in range(coords.shape[0]):
        coord = coords[i, :]

        # Reference
        expected = reference_hc.distance_from_point(coord)

        # Compute scalar distance and compare
        scalar_result = distance_from_coordinate(p, coord)
        assert scalar_result == expected

        # Compare with vector distance compute above
        assert vector_result[i] == expected
