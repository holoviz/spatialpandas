import pandas.tests.extension.base as eb
import pytest
from spatialpandas.geometry import LineArray, LineDtype


def test_equality():
    a = LineArray([[0, 1], [1, 2, 3, 4], None, [-1, -2], []], dtype='float64')
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
    return LineDtype(subtype='float64')


@pytest.fixture
def data():
    """Length-100 array for this type.
        * data[0] and data[1] should both be non missing
        * data[0] and data[1] should not gbe equal
        """
    return LineArray(
        [[0, 1], [1, 2, 3, 4], None, [-1, -2], []]*20, dtype='float64')


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
    return LineArray([None, [-1, 0, 1, 2]], dtype='int64')


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
    return LineArray([[1, 0], [2, 0], [0, 0]])


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.
    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return LineArray([[1, 0], None, [0, 0]])


@pytest.fixture
def data_for_grouping():
    """Data for factorization, grouping, and unique tests.
    Expected to be like [B, B, NA, NA, A, A, B, C]
    Where A < B < C and NA is missing
    """
    return LineArray(
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


class TestGeometryGroupby(eb.BaseGroupbyTests):
    pass


class TestGeometryInterface(eb.BaseInterfaceTests):
    # # NotImplementedError: 'GeometryList' does not support __setitem__
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_copy(self):
        pass


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


class TestGeometryPrinting(eb.BasePrintingTests):
    pass


class TestGeometryMissing(eb.BaseMissingTests):
    pass


class TestGeometryReshaping(eb.BaseReshapingTests):
    @pytest.mark.skip(reason="__setitem__ not supported")
    def test_ravel(self):
        pass

