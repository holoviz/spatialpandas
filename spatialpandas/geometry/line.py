from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry._algorithms import compute_line_length, \
    geometry_map_nested1
from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry0
)
import numpy as np
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class LineDtype(GeometryDtype):
    _geometry_name = 'line'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return LineArray


class Line(Geometry0):
    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, (sg.LineString, sg.LinearRing)):
            # Single line
            return np.asarray(shape.ctypes)
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of LineString
or LinearRing""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely LineString shape
        """
        import shapely.geometry as sg
        line_coords = self.data.to_numpy()
        return sg.LineString(line_coords.reshape(len(line_coords) // 2, 2))

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas Line object from a shapely shape

        Args:
            shape: A shapely LineString or LinearRing shape

        Returns:
            spatialpandas Line
        """
        return super(Line, cls).from_shapely(shape)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return 0.0


class LineArray(GeometryArray):
    _element_type = Line
    _nesting_levels = 1

    @property
    def _dtype_class(self):
        return LineDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas LineArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of `LineString` or
            `LinearRing`shapes.

        Returns:
            LineArray
        """
        return super(LineArray, cls).from_geopandas(ga)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        for c, result_offset in enumerate(self.offsets):
            geometry_map_nested1(
                compute_line_length,
                result,
                result_offset,
                self.buffer_values,
                self.buffer_offsets,
                self.isna(),
            )
        return result

    @property
    def area(self):
        return np.zeros(len(self), dtype=np.float64)


def _line_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return LineArray([
        [1, 0, 1, 1],
        [1, 2, 0, 0]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(LineDtype)(_line_array_non_empty)
