from hypothesis import given
import hypothesis.strategies as st
from hypothesis import settings
from hypothesis.extra.numpy import arrays
import numpy as np
from spatialpandas.spatialindex import HilbertRtree
import pickle

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
def rtree_intersection(rt, bounds):
    return set(rt.intersection(np.array(bounds)))


def intersects_bruteforce(query_bounds, bounds):
    if len(bounds) == 0:
        return set()

    outside_mask = np.zeros(bounds.shape[0], dtype=np.bool_)
    n = bounds.shape[1] // 2
    for d in range(n):
        outside_mask |= (bounds[:, d + n] < query_bounds[d])
        outside_mask |= (bounds[:, d] > query_bounds[d + n])

    return set(np.nonzero(~outside_mask)[0])


# ### Hypothesis tests ###
@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_query_input_bounds(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with bounds array, these were the input
    for i in range(bounds_array.shape[0]):
        b = bounds_array[i, :]
        intersected = set(rt.intersection(np.array(b)))

        # Should at least select itself
        assert i in intersected

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(
    st_bounds_array(n_min=2, n_max=2),
    st_bounds_array(n_min=2, n_max=2, min_size=10),
    st_page_size
)
@hyp_settings
def test_rtree_query_different_bounds_2d(bounds_array, query_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    for i in range(query_array.shape[0]):
        b = query_array[i, :]
        intersected = set(rt.intersection(np.array(b)))

        # Compare to brute force implementation
        assert intersected == intersects_bruteforce(b, bounds_array)


@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_query_all(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    n = bounds_array.shape[1] // 2
    b_mins = bounds_array[:, :n].min(axis=0)
    b_maxes = bounds_array[:, n:].max(axis=0)
    b = np.concatenate([b_mins, b_maxes])
    intersected = set(rt.intersection(np.array(b)))

    # All indices should be selected
    assert intersected == set(range(bounds_array.shape[0]))


@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_query_none(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # query with query array
    n = bounds_array.shape[1] // 2
    b = ([11] * n) + ([100] * n)
    intersected = set(rt.intersection(np.array(b)))

    # Nothing should be selected
    assert intersected == set()


@given(st_bounds_array(), st_page_size)
@hyp_settings
def test_rtree_query_input_bounds_pickle(bounds_array, page_size):
    # Build rtree
    rt = HilbertRtree(bounds_array, page_size=page_size)

    # Serialize uninitialized rtree
    rt2 = pickle.loads(pickle.dumps(rt))
    rt2_result = set(rt2.intersection(bounds_array[0, :]))

    # Call intersection to construct numba rtree
    rt_result = set(rt.intersection(bounds_array[0, :]))

    # Serialize initialized rtree
    rt3 = pickle.loads(pickle.dumps(rt))
    rt3_result = set(rt3.intersection(bounds_array[0, :]))
    assert rt_result == rt2_result and rt_result == rt3_result
