import pickle

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from spatialpandas.spatialindex import HilbertRtree

# ### hypothesis settings ###
hyp_settings = settings(deadline=None)


# ### Custom strategies ###

@st.composite
def st_bounds(draw, n_min=1, n_max=3):
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    dim_starts = [draw(st.floats(0, 10**(i + 1) - 1)) for i in range(n)]
    dim_widths = [draw(st.floats(0, 10 ** i)) for i in range(n)]
    dim_ends = [s + w for s, w in zip(dim_starts, dim_widths)]
    return tuple(dim_starts + dim_ends)


@st.composite
def st_bounds_array(draw, n_min=1, n_max=3, min_size=1, max_size=1000):
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    size = draw(st.integers(min_value=min_size, max_value=max_size))

    dim_starts = draw(
        arrays(elements=st.floats(0, 9), shape=(size, n), dtype='float64')
    )
    dim_widths = draw(
        arrays(elements=st.floats(0, 1), shape=(size, n), dtype='float64')
    )
    dim_ends = dim_starts + dim_widths
    bounds_array = np.concatenate([dim_starts, dim_ends], axis=1)
    if bounds_array.size == 0:
        bounds_array = bounds_array.reshape(0, 2*n)
    return bounds_array


st_page_size = st.integers(min_value=-1, max_value=1000000)

# ### Test custom strategies ###
@given(st_bounds())
def test_bounds_generation(bounds):
    assert len(bounds) % 2 == 0
    n = len(bounds) // 2
    for d in range(n):
        assert bounds[d + n] >= bounds[d]


@given(st_bounds_array())
def test_bounds_array_generation(bounds_array):
    assert bounds_array.shape[1] % 2 == 0
    n = bounds_array.shape[1] // 2
    for d in range(n):
        assert np.all(bounds_array[:, d + n] >= bounds_array[:, d])


# ### utilities ###
def intersects_bruteforce(query_bounds, bounds):
    if len(bounds) == 0:
        return set()

    outside_mask = np.zeros(bounds.shape[0], dtype=np.bool_)
    n = bounds.shape[1] // 2
    for d in range(n):
        outside_mask |= (bounds[:, d + n] < query_bounds[d])
        outside_mask |= (bounds[:, d] > query_bounds[d + n])

    return set(np.nonzero(~outside_mask)[0])


def covers_bruteforce(query_bounds, bounds):
    if len(bounds) == 0:
        return set()

    not_covers_mask = np.zeros(bounds.shape[0], dtype=np.bool_)
    n = bounds.shape[1] // 2
    for d in range(n):
        not_covers_mask |= (bounds[:, d] < query_bounds[d])
        not_covers_mask |= (bounds[:, d + n] > query_bounds[d + n])

    return set(np.nonzero(~not_covers_mask)[0])


# ### Hypothesis tests ###
@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_intersects_input_bounds(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with bounds array, these were the input
    for i in range(bounds_array.shape[0]):
        b = bounds_array[i, :]
        intersected = set(rt.intersects(np.array(b)))

        # Should at least select itself
        assert i in intersected

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(st_bounds_array(min_size=0, max_size=0), st_page_size)
@hyp_settings
def test_rtree_empty(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)
    n = bounds_array.shape[1] // 2
    query_bounds = (-np.inf,) * n + (np.inf,) * n
    assert set(rt.intersects(query_bounds)) == set()
    covers, overlaps = rt.covers_overlaps(query_bounds)
    assert set(covers) == set()
    assert set(overlaps) == set()
    assert rt.total_bounds == (np.nan,) * (2 * n)


@given(
    st_bounds_array(n_min=2, n_max=2),
    st_bounds_array(n_min=2, n_max=2, min_size=10),
    st_page_size
)
@hyp_settings
def test_rtree_intersects_different_bounds_2d(bounds_array, query_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    for i in range(query_array.shape[0]):
        b = query_array[i, :]
        intersected = set(rt.intersects(np.array(b)))

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(st_bounds_array(n_min=1), st_page_size)
@hyp_settings
def test_rtree_intersects_all(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    n = bounds_array.shape[1] // 2
    b_mins = bounds_array[:, :n].min(axis=0)
    b_maxes = bounds_array[:, n:].max(axis=0)
    b = np.concatenate([b_mins, b_maxes])
    intersected = set(rt.intersects(np.array(b)))

    # All indices should be selected
    assert intersected == set(range(bounds_array.shape[0]))


@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_intersects_none(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    n = bounds_array.shape[1] // 2
    b = ([11] * n) + ([100] * n)
    intersected = set(rt.intersects(np.array(b)))

    # Nothing should be selected
    assert intersected == set()


@pytest.mark.slow
@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_covers_overlaps_input_bounds(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with bounds array, these were the input
    for i in range(bounds_array.shape[0]):
        b = bounds_array[i, :]
        covers, overlaps = rt.covers_overlaps(np.array(b))
        covers = set(covers)
        overlaps = set(overlaps)

        # Should have nothing in common
        assert covers.isdisjoint(overlaps)

        # covered should contain all bounding boxes that are fully covered by query
        all_covers = covers_bruteforce(b, bounds_array)
        if covers != all_covers:
            rt.covers_overlaps(np.array(b))
        assert covers == all_covers

        # covered and overlaps together should contain all intersecting bounding
        # boxes (overlaps will typically contain others as well)
        intersects = intersects_bruteforce(b, bounds_array)
        assert intersects == covers.union(overlaps)


@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_pickle(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # Serialize uninitialized rtree
    rt2 = pickle.loads(pickle.dumps(rt))
    if bounds_array.size == 0:
        assert isinstance(rt2, HilbertRtree)
        return

    rt2_result = set(rt2.intersects(bounds_array[0, :]))

    # Call intersection to construct numba rtree
    rt_result = set(rt.intersects(bounds_array[0, :]))

    # Serialize initialized rtree
    rt3 = pickle.loads(pickle.dumps(rt))
    rt3_result = set(rt3.intersects(bounds_array[0, :]))
    assert rt_result == rt2_result and rt_result == rt3_result
