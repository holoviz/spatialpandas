from hypothesis import given
import hypothesis.strategies as st
from hypothesis import settings
import numpy as np
from spatialpandas.spatialindex.rtree import HilbertRtree

# ### hypothesis settings ###
hyp_settings = settings(deadline=None)


# ### Custom strategies ###
@st.composite
def st_bounds(draw):
    x0 = draw(st.floats(0, 9))
    dx = draw(st.floats(0, 1))
    y0 = draw(st.floats(0, 90))
    dy = draw(st.floats(0, 10))
    return x0, y0, x0 + dx, y0 + dy


@st.composite
def st_bounds_array(draw, min_size=1, max_size=None):
    return np.array(draw(
        st.lists(st_bounds(), min_size=min_size, max_size=max_size)
    ))


# ### Test custom strategies ###
@given(st_bounds())
def test_bounds_generation(bounds):
    assert len(bounds) == 4
    assert bounds[2] >= bounds[0]
    assert bounds[3] >= bounds[1]


# ### utilities ###
def build_rtree(bounds_array):
    return HilbertRtree(bounds_array)


def rtree_intersection(rt, bounds):
    result = np.zeros(rt.input_size, dtype='int64')
    end_ind = rt.intersection(np.array(bounds), result)
    return set(result[:end_ind])


def intersects_bruteforce(query_bounds, bounds):
    if len(bounds) == 0:
        return set()

    x0, y0, x1, y1 = query_bounds
    outside = ((bounds[:, 2] < x0) | (bounds[:, 0] > x1) |
               (bounds[:, 3] < y0) | (bounds[:, 1] > y1))
    return set(np.nonzero(~outside)[0])


# ### Hypothesis tests ###
@given(st_bounds_array())
@hyp_settings
def test_rtree_query_input_bounds(bounds_array):
    # Build rtree
    rt = build_rtree(bounds_array)

    # query with bounds array, these were the input
    for i in range(bounds_array.shape[0]):
        b = bounds_array[i, :]
        intersected = rtree_intersection(rt, b)

        # Should at least select itself
        assert i in intersected

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(st_bounds_array(), st_bounds_array(min_size=10))
@hyp_settings
def test_rtree_query_different_bounds(bounds_array, query_array):
    # Build rtree
    rt = build_rtree(bounds_array)

    # query with query array
    for i in range(query_array.shape[0]):
        b = query_array[i, :]
        intersected = rtree_intersection(rt, b)

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(st_bounds_array())
@hyp_settings
def test_rtree_query_all(bounds_array):
    # Build rtree
    rt = build_rtree(bounds_array)

    # query with query array
    b = [bounds_array[:, 0].min(), bounds_array[:, 1].min(),
         bounds_array[:, 2].max(), bounds_array[:, 3].max()]
    intersected = rtree_intersection(rt, b)

    # All indices should be selected
    assert intersected == set(range(bounds_array.shape[0]))


@given(st_bounds_array())
@hyp_settings
def test_rtree_query_none(bounds_array):
    # Build rtree
    rt = build_rtree(bounds_array)

    # query with query array
    b = [11, 101, 20, 110]
    intersected = rtree_intersection(rt, b)

    # Nothing should be selected
    assert intersected == set()
