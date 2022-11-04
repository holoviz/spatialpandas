import pandas.tests.extension.base as eb
import pytest

from spatialpandas.geometry import PointArray, PointDtype


def test_equality():
    a = PointArray([[0, 1], [1, 2], None, [-1, -2], [7, 3]], dtype='float64')
    assert all(a == a)
    assert all(a[1:-1] == a[[1, 2, 3]])
    assert not any(a[1:-1] == a[[2, 3, 1]])


# Pandas-provided extension array tests
# -------------------------------------
# See http://pandas-docs.github.io/pandas-docs-travis/extending.html
#
# Fixtures
@pytest.fixture
def dtype():
    """A fixture providing the ExtensionDtype to validate."""
    return PointDtype(subtype='float64')


@pytest.fixture
def data():
    """Length-100 array for this type.
        * data[0] and data[1] should both be non missing
        * data[0] and data[1] should not gbe equal
        """
    return PointArray(
        [[0, 1], [1, 2], [3, 4], None, [-1, -2]]*20, dtype='float64')


@pytest.fixture
def data_repeated(data):
    """
    Generate many datasets.
    Parameters
    ----------
    data : fixture implementing `data`
    Returns
    -------
    Callable[[int], Generator]:
        A callable that takes a `count` argument and
        returns a generator yielding `count` datasets.
    """
    def gen(count):
        for _ in range(count):
            yield data
    return gen


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return PointArray([None, [-1, 0]], dtype='int64')


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, C, A] with
    A < B < C
    """
    return PointArray([[1, 0], [2, 0], [0, 0]])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return PointArray([[1, 0], None, [0, 0]])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    return PointArray(
        [[1, 0], [1, 0], None, None, [0, 0], [0, 0], [1, 0], [2, 0]])


@pytest.fixture
def na_cmp():
    return lambda x, y: x is None and y is None


@pytest.fixture
def na_value():
    return None


@pytest.fixture
def groupby_apply_op():
    return lambda x: [1] * len(x)


@pytest.fixture
def fillna_method():
    return 'ffill'


@pytest.fixture(params=[True, False])
def as_frame(request):
    return request.param


@pytest.fixture(params=[True, False])
def as_series(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy(request):
    return request.param


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    copied from pandas.conftest. Not sure why importing this module isn't enough
    to register fixture with pytest

    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


# Subclass BaseDtypeTests to run pandas-provided extension array test suite
class TestGeometryConstructors(eb.BaseConstructorsTests):
    only_run_named_tests = set([
        "test_array_from_scalars",
        "test_construct_empty_dataframe",
        "test_dataframe_constructor_from_dict",
        "test_dataframe_from_series",
        "test_empty",
        "test_from_dtype",
        "test_from_sequence_from_cls",
        "test_pandas_array",
        "test_pandas_array_dtype",
        "test_series_constructor",
        "test_series_constructor_no_data_with_index",
        "test_series_constructor_scalar_na_with_index",
        "test_series_constructor_scalar_with_index",
        "test_series_given_mismatched_index_raises",
    ])


class TestGeometryDtype(eb.BaseDtypeTests):
    only_run_named_tests = set([
        "test_array_type",
        "test_check_dtype",
        "test_construct_from_string",
        "test_construct_from_string_another_type_raises",
        "test_construct_from_string_own_name",
        "test_construct_from_string_wrong_type_raises",
        "test_eq",
        "test_eq_with_numpy_object",
        "test_eq_with_self",
        "test_eq_with_str",
        "test_get_common_dtype",
        "test_hashable",
        "test_infer_dtype",
        "test_is_dtype_from_name",
        "test_is_dtype_from_self",
        "test_is_dtype_other_input",
        "test_is_dtype_unboxes_dtype",
        "test_is_not_object_type",
        "test_is_not_string_type",
        "test_kind",
        "test_name",
        "test_str",
    ])


class TestGeometryGetitem(eb.BaseGetitemTests):
    only_run_named_tests = set([
        "test_ellipsis_index",
        "test_get",
        "test_getitem_boolean_array_mask",
        #"test_getitem_boolean_na_treated_as_false",
        "test_getitem_ellipsis_and_slice",
        "test_getitem_empty",
        "test_getitem_integer_array",
        "test_getitem_integer_with_missing_raises",
        #"test_getitem_invalid",
        "test_getitem_mask",
        "test_getitem_mask_raises",
        "test_getitem_scalar",
        "test_getitem_scalar_na",
        "test_getitem_series_integer_with_missing_raises",
        "test_getitem_slice",
        "test_iloc_frame",
        "test_iloc_frame_single_block",
        "test_iloc_series",
        "test_item",
        "test_loc_frame",
        "test_loc_iloc_frame_single_dtype",
        "test_loc_len1",
        "test_loc_series",
        "test_reindex",
        #"test_reindex_non_na_fill_value",
        "test_take",
        "test_take_empty",
        "test_take_negative",
        #"test_take_non_na_fill_value",
        "test_take_out_of_bounds_raises",
        "test_take_pandas_style_negative_raises",
        "test_take_sequence",
        "test_take_series",
    ])


class TestGeometryGroupby(eb.BaseGroupbyTests):
    only_run_named_tests = set([
        "test_groupby_agg_extension",
        "test_groupby_apply_identity",
        "test_groupby_extension_agg",
        "test_groupby_extension_apply",
        "test_groupby_extension_no_sort",
        "test_groupby_extension_transform",
        "test_grouping_grouper",
        "test_in_numeric_groupby",
    ])


class TestGeometryInterface(eb.BaseInterfaceTests):
    only_run_named_tests = set([
        "test_array_interface",
        "test_can_hold_na_valid",
        #"test_contains",
        #"test_copy",
        "test_is_extension_array_dtype",
        "test_isna_extension_array",
        "test_is_numeric_honored",
        "test_len",
        "test_memory_usage",
        "test_ndim",
        "test_no_values_attribute",
        "test_size",
        "test_tolist",
        #"test_view",
    ])


class TestGeometryMethods(eb.BaseMethodsTests):
    only_run_named_tests = set([
        "test_apply_simple_series",
        "test_argmax_argmin_no_skipna_notimplemented",
        "test_argmin_argmax",
        "test_argmin_argmax_all_na",
        "test_argmin_argmax_empty_array",
        "test_argreduce_series",
        "test_argsort",
        "test_argsort_missing",
        "test_argsort_missing_array",
        #"test_combine_add",
        #"test_combine_first",
        #"test_combine_le",
        "test_container_shift",
        "test_count",
        "test_delete",
        "test_diff",
        "test_equals",
        "test_factorize",
        "test_factorize_empty",
        "test_factorize_equivalence",
        "test_fillna_copy_frame",
        "test_fillna_copy_series",
        "test_fillna_length_mismatch",
        "test_hash_pandas_object_works",
        "test_insert",
        #"test_insert_invalid",
        "test_insert_invalid_loc",
        "test_nargsort",
        "test_not_hashable",
        "test_repeat",
        "test_repeat_raises",
        #"test_searchsorted",
        "test_series_count",
        #"test_shift_0_periods",
        "test_shift_empty_array",
        "test_shift_fill_value",
        "test_shift_non_empty_array",
        "test_shift_zero_copies",
        "test_sort_values",
        "test_sort_values_frame",
        "test_sort_values_missing",
        "test_unique",
        #"test_value_counts",
        "test_value_counts_default_dropna",
        #"test_value_counts_with_normalize",
        #"test_where_series",
    ])


class TestGeometryPrinting(eb.BasePrintingTests):
    only_run_named_tests = set([
        "test_array_repr",
        "test_array_repr_unicode",
        "test_dataframe_repr",
        "test_dtype_name_in_info",
        "test_series_repr",
    ])


class TestGeometryMissing(eb.BaseMissingTests):
    only_run_named_tests = set([
        "test_dropna_array",
        "test_dropna_frame",
        "test_dropna_series",
        "test_fillna_fill_other",
        "test_fillna_frame",
        "test_fillna_limit_backfill",
        "test_fillna_limit_pad",
        "test_fillna_no_op_returns_copy",
        "test_fillna_scalar",
        "test_fillna_series",
        "test_fillna_series_method",
        "test_isna",
        "test_isna_returns_copy",
        "test_use_inf_as_na_no_effect",
    ])


class TestGeometryReshaping(eb.BaseReshapingTests):
    only_run_named_tests = set([
        "test_align",
        "test_align_frame",
        "test_align_series_frame",
        "test_concat",
        "test_concat_all_na_block",
        "test_concat_columns",
        "test_concat_extension_arrays_copy_false",
        "test_concat_mixed_dtypes",
        "test_concat_with_reindex",
        "test_merge",
        "test_merge_on_extension_array",
        "test_merge_on_extension_array_duplicates",
        #"test_ravel",
        "test_set_frame_expand_extension_with_regular",
        "test_set_frame_expand_regular_with_extension",
        "test_set_frame_overwrite_object",
        "test_stack",
        #"test_transpose",
        "test_transpose_frame",
        "test_unstack",
    ])
