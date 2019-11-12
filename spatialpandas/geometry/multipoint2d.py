import numpy as np
from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry0
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class MultiPoint2dDtype(GeometryDtype):
    _geometry_name = 'multipoint2d'
    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiPoint2dArray


class MultiPoint2d(Geometry0):

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.Point, sg.MultiPoint)):
            # Single line
            return np.asarray(shape.ctypes)
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Point,
or MultiPoint2d""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely MultiPoint shape
        """
        import shapely.geometry as sg
        line_coords = self.data.to_numpy()
        return sg.MultiPoint(line_coords.reshape(len(line_coords) // 2, 2))

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas MultiPoint2d object from a shapely shape

        Args:
            shape: A shapely MultiPoint or Point shape

        Returns:
            spatialpandas MultiPoint2d
        """
        return super(MultiPoint2d, cls).from_shapely(shape)

    @property
    def length(self):
        return 0.0

    @property
    def area(self):
        return 0.0


class MultiPoint2dArray(GeometryArray):
    _element_type = MultiPoint2d
    _nesting_levels = 1

    @property
    def _dtype_class(self):
        return MultiPoint2dDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas MultiPoint2dArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiPoint or
            Point shapes.

        Returns:
            MultiPoint2dArray
        """
        return super(MultiPoint2dArray, cls).from_geopandas(ga)

    @property
    def length(self):
        return np.zeros(len(self), dtype=np.float64)

    @property
    def area(self):
        return np.zeros(len(self), dtype=np.float64)


def _multi_points_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return MultiPoint2dArray([
        [1, 0, 1, 1],
        [1, 2, 0, 0]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(MultiPoint2dDtype)(_multi_points_array_non_empty)
