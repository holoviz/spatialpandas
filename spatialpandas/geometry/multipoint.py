import numpy as np
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry._algorithms.intersection import multipoints_intersect_bounds
from ..geometry.base import GeometryDtype
from ..geometry.baselist import GeometryList, GeometryListArray


@register_extension_dtype
class MultiPointDtype(GeometryDtype):
    _geometry_name = 'multipoint'
    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiPointArray


class MultiPoint(GeometryList):
    _nesting_levels = 0

    @classmethod
    def construct_array_type(cls):
        return MultiPointArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.Point, sg.MultiPoint)):
            # Single line
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
        point_coords = np.array(self.data.as_py(), dtype=self.numpy_dtype)
        return sg.MultiPoint(point_coords.reshape(len(point_coords) // 2, 2))

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas MultiPoint object from a shapely shape

        Args:
            shape: A shapely MultiPoint or Point shape

        Returns:
            spatialpandas MultiPoint
        """
        return super().from_shapely(shape)

    @property
    def length(self):
        return 0.0

    @property
    def area(self):
        return 0.0

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        result = np.zeros(1, dtype=np.bool_)
        offsets = self.buffer_outer_offsets
        multipoints_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, offsets[:-1], offsets[1:], result
        )
        return result[0]


class MultiPointArray(GeometryListArray):
    _element_type = MultiPoint
    _nesting_levels = 1

    @property
    def _dtype_class(self):
        return MultiPointDtype

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

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        offsets0 = self.buffer_outer_offsets
        start_offsets0 = offsets0[:-1]
        stop_offsets0 = offsets0[1:]

        if inds is not None:
            start_offsets0 = start_offsets0[inds]
            stop_offsets0 = stop_offsets0[inds]

        result = np.zeros(len(start_offsets0), dtype=np.bool_)
        multipoints_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets0, stop_offsets0, result
        )
        return result


def _multi_points_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return MultiPointArray([
        [1, 0, 1, 1],
        [1, 2, 0, 0]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(MultiPointDtype)(_multi_points_array_non_empty)
