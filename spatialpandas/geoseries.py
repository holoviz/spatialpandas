import pandas as pd
from packaging.version import Version

from .geometry import GeometryDtype, Geometry


class _MaybeGeoSeries(pd.Series):
    def __new__(cls, data, *args, **kwargs):
        if isinstance(getattr(data, 'dtype', None), GeometryDtype):
            series_cls = GeoSeries
        else:
            series_cls = pd.Series
        return series_cls(data, *args, **kwargs)


class GeoSeries(pd.Series):
    def __init__(self, data, index=None, name=None, dtype=None, **kwargs):
        from .geometry.base import to_geometry_array

        # Handle scalar geometry with index
        if isinstance(data, Geometry):
            n = len(index) if index is not None else 1
            data = [data] * n

        if index is None and hasattr(data, 'dtype'):
            # Try to get input index from Series-like object
            index = getattr(data, 'index', None)
        if name is None:
            name = getattr(data, 'name', None)

        # Normalize dtype from string
        if dtype is not None:
            dtype = pd.array([], dtype=dtype).dtype

        data = to_geometry_array(data, dtype)
        super().__init__(data, index=index, name=name, **kwargs)

    def _constructor_from_mgr(self, mgr, axes):
        if Version(pd.__version__) < Version('2.1'):
            return super()._constructor_from_mgr(mgr, axes)

        # Copied from pandas.Series._constructor_from_mgr
        # https://github.com/pandas-dev/pandas/blob/80a1a8bc3e07972376284ffce425a2abd1e58604/pandas/core/series.py#L582-L590
        # Changed if statement to use GeoSeries instead of Series.
        # REF: https://github.com/pandas-dev/pandas/pull/52132
        # REF: https://github.com/pandas-dev/pandas/pull/53871

        ser = self._from_mgr(mgr, axes=axes)
        ser._name = None  # caller is responsible for setting real name

        if type(self) is GeoSeries:  # Changed line
            # fastpath avoiding constructor call
            return ser
        else:
            assert axes is mgr.axes
            return self._constructor(ser, copy=False)

    @property
    def _constructor(self):
        return _MaybeGeoSeries

    @property
    def _constructor_expanddim(self):
        from .geodataframe import GeoDataFrame
        return GeoDataFrame

    @property
    def bounds(self):
        return pd.DataFrame(
            self.array.bounds, columns=['x0', 'y0', 'x1', 'y1'], index=self.index
        )

    @property
    def total_bounds(self):
        return self.array.total_bounds

    @property
    def area(self):
        return pd.Series(self.array.area, index=self.index)

    @property
    def length(self):
        return pd.Series(self.array.length, index=self.index)

    def hilbert_distance(self, total_bounds=None, p=15):
        return pd.Series(
            self.array.hilbert_distance(total_bounds=total_bounds, p=p),
            index=self.index
        )

    @property
    def sindex(self):
        return self.array.sindex

    @property
    def cx(self):
        from .geometry.base import _CoordinateIndexer
        return _CoordinateIndexer(self.array, parent=self)

    def build_sindex(self, **kwargs):
        self.array.build_sindex(**kwargs)
        return self

    def intersects_bounds(self, bounds):
        return pd.Series(
            self.array.intersects_bounds(bounds), index=self.index
        )

    def intersects(self, shape):
        return pd.Series(
            self.array.intersects(shape), index=self.index
        )

    def to_geopandas(self):
        from geopandas import GeoSeries
        return GeoSeries(self.array.to_geopandas(), index=self.index)
