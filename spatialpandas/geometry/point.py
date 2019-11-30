import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.base import GeometryDtype
from spatialpandas.geometry.basefixed import GeometryFixed, GeometryFixedArray
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class PointDtype(GeometryDtype):
    _geometry_name = 'point'
    @classmethod
    def construct_array_type(cls, *args):
        return PointArray


class Point(GeometryFixed):

    @classmethod
    def construct_array_type(cls):
        return PointArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.Point):
            # Single point
            return np.asarray(shape.ctypes)
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Point,
or MultiPoint""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely MultiPoint shape
        """
        import shapely.geometry as sg
        return sg.Point(self.flat_values)

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas Point object from a shapely shape

        Args:
            shape: A shapely MultiPoint or Point shape

        Returns:
            spatialpandas MultiPoint
        """
        return super().from_shapely(shape)

    @property
    def x(self):
        return self.flat_values[0]

    @property
    def y(self):
        return self.flat_values[1]

    @property
    def length(self):
        return 0.0

    @property
    def area(self):
        return 0.0

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        outside = (np.isnan(self.x) or
                   self.x < x0 or self.x > x1 or
                   self.y < y0 or self.y > y1)

        return not outside


class PointArray(GeometryFixedArray):
    _element_type = Point

    @property
    def _dtype_class(self):
        return PointDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas MultiPointArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiPoint or
            Point shapes.

        Returns:
            MultiPointArray
        """
        return super().from_geopandas(ga)

    @property
    def length(self):
        return np.zeros(len(self), dtype=np.float64)

    @property
    def area(self):
        return np.zeros(len(self), dtype=np.float64)

    @property
    def x(self):
        res = np.full(len(self), np.nan)
        mask = ~self.isna()
        res[mask] = self.flat_values[0::2][mask]
        return res

    @property
    def y(self):
        res = np.full(len(self), np.nan)
        mask = ~self.isna()
        res[mask] = self.flat_values[1::2][mask]
        return res

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        xs = self.x
        ys = self.y
        if inds is not None:
            xs = xs[inds]
            ys = ys[inds]

        outside = np.isnan(xs) | (xs < x0) | (xs > x1) | (ys < y0) | (ys > y1)
        return ~outside


def _points_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return PointArray([
        [1, 0],
        [1, 2]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(PointDtype)(_points_array_non_empty)
