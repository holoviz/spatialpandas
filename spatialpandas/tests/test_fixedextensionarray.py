import itertools

import pandas as pd
import pandas.tests.extension.base as eb
import pytest

from spatialpandas.geometry import PointArray, PointDtype


def test_equality():
    a = PointArray([[0, 1], [1, 2], None, [-1, -2], [7, 3]], dtype='float64')
    assert all(a == a)  # noqa: PLR0124
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
    pass


class TestGeometryDtype(eb.BaseDtypeTests):
    pass


class TestGeometryGetitem(eb.BaseGetitemTests):
    @pytest.mark.skip(reason="non-None fill value not supported")
    def test_take_non_na_fill_value(self):
        pass

    @pytest.mark.skip(reason="non-None fill value not supported")
    def test_reindex_non_na_fill_value(self, data_missing):
        pass

    @pytest.mark.skip("Cannot mask with a boolean indexer containing NA values")
    def test_getitem_boolean_na_treated_as_false(self, data):
        pass

    @pytest.mark.skip("Passing an invalid index type is not supported")
    def test_getitem_invalid(self, data):
        pass

    @pytest.mark.skip("__getitem__ with keys as positions is deprecated")
    def test_getitem_series_integer_with_missing_raises(self, data, idx):
        pass

    @pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
    def test_take_pandas_style_negative_raises(self, data, na_value):
        super().test_take_pandas_style_negative_raises(data, na_value)


class TestGeometryGroupby(eb.BaseGroupbyTests):
    @pytest.mark.skip(
        reason="The truth value of an array with more than one element is ambiguous."
    )
    def test_groupby_apply_identity(self):
        pass


class TestGeometryInterface(eb.BaseInterfaceTests):
    # # NotImplementedError: 'GeometryList' does not support __setitem__
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_copy(self):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_view(self, data):
        pass

    @pytest.mark.skip(reason="contains not supported")
    def test_contains(self):
        pass

    @pytest.mark.xfail(reason="copy=False not supported")
    def test_array_interface_copy(self, data):
        super().test_array_interface_copy(data)


class TestGeometryMethods(eb.BaseMethodsTests):
    # # AttributeError: 'RaggedArray' object has no attribute 'value_counts'
    @pytest.mark.skip(reason="value_counts not supported")
    def test_value_counts(self):
        pass

    # Ragged array elements don't support binary operators
    @pytest.mark.skip(reason="ragged does not support <= on elements")
    def test_combine_le(self):
        pass

    @pytest.mark.skip(reason="ragged does not support + on elements")
    def test_combine_add(self):
        pass

    @pytest.mark.skip(reason="combine_first not supported")
    def test_combine_first(self):
        pass

    @pytest.mark.skip(reason="ragged does not support insert with an invalid scalar")
    def test_insert_invalid(self, data, invalid_scalar):
        pass

    @pytest.mark.skip(
        reason="Searchsorted seems not implemented for custom extension arrays"
    )
    def test_searchsorted(self):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_fillna_copy_frame(self, data):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_fillna_copy_series(self, data):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_shift_0_periods(self, data):
        pass

    @pytest.mark.skip(
        reason="value_counts not yet supported"
    )
    def test_value_counts_with_normalize(self, data):
        pass

    @pytest.mark.skip(reason="ragged does not support where on elements")
    def test_where_series(self):
        pass

    @pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
    def test_argmax_argmin_no_skipna_notimplemented(self, data_missing_for_sorting):
        super().test_argmax_argmin_no_skipna_notimplemented(data_missing_for_sorting)


class TestGeometryPrinting(eb.BasePrintingTests):
    pass


class TestGeometryMissing(eb.BaseMissingTests):
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_fillna_frame(self):
        pass

    @pytest.mark.skip(reason="not implemented")
    def test_fillna_limit_backfill(self):
        pass

    @pytest.mark.skip(reason="not implemented")
    def test_fillna_limit_pad(self):
        pass

    @pytest.mark.skip(reason="not implemented")
    def test_fillna_no_op_returns_copy(self):
        pass

    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_fillna_series(self):
        pass

    @pytest.mark.skip(reason="not implemented")
    def test_fillna_series_method(self):
        pass

    @pytest.mark.skip(reason="not implemented")
    def test_ffill_limit_area(self):
        pass

class TestGeometryReshaping(eb.BaseReshapingTests):
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_ravel(self):
        pass

    @pytest.mark.skip(reason="transpose with numpy array elements seems not supported")
    def test_transpose(self):
        pass

    # NOTE: this test is copied from pandas/tests/extension/base/reshaping.py
    # because starting with pandas 3.0 the assert_frame_equal is strict regarding
    # the exact missing value (None vs NaN)
    # Our `result` uses None, but the way the `expected` is created results in
    # NaNs (and specifying to use None as fill value in unstack also does not
    # help)
    # -> the only real change compared to the upstream test is marked
    # Change from: https://github.com/geopandas/geopandas/pull/3234
    @pytest.mark.parametrize(
            "index",
        [
            # Two levels, uniform.
            pd.MultiIndex.from_product(([["A", "B"], ["a", "b"]]), names=["a", "b"]),
            # non-uniform
            pd.MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "b")]),
            # three levels, non-uniform
            pd.MultiIndex.from_product([("A", "B"), ("a", "b", "c"), (0, 1, 2)]),
            pd.MultiIndex.from_tuples(
                [
                    ("A", "a", 1),
                    ("A", "b", 0),
                    ("A", "a", 0),
                    ("B", "a", 0),
                    ("B", "c", 1),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("obj", ["series", "frame"])
    def test_unstack(self, data, index, obj):
        data = data[: len(index)]
        if obj == "series":
            ser = pd.Series(data, index=index)
        else:
            ser = pd.DataFrame({"A": data, "B": data}, index=index)

        n = index.nlevels
        levels = list(range(n))
        # [0, 1, 2]
        # [(0,), (1,), (2,), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        combinations = itertools.chain.from_iterable(
            itertools.permutations(levels, i) for i in range(1, n)
        )

        for level in combinations:
            result = ser.unstack(level=level)
            assert all(
                isinstance(result[col].array, type(data)) for col in result.columns
            )

            if obj == "series":
                # We should get the same result with to_frame+unstack+droplevel
                df = ser.to_frame()

                alt = df.unstack(level=level).droplevel(0, axis=1)
                pd.testing.assert_frame_equal(result, alt)

            obj_ser = ser.astype(object)

            expected = obj_ser.unstack(level=level, fill_value=data.dtype.na_value)
            if obj == "series":
                assert (expected.dtypes == object).all()  # noqa: E721
            # <------------ next line is added
            expected[expected.isna()] = None
            # ------------->

            result = result.astype(object)
            pd.testing.assert_frame_equal(result, expected)
