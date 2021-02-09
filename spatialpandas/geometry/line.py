import numpy as np
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry._algorithms.intersection import lines_intersect_bounds
from ..geometry._algorithms.measures import compute_line_length
from ..geometry.base import GeometryDtype
from ..geometry.baselist import (GeometryList, GeometryListArray,
                                 _geometry_map_nested1)


@register_extension_dtype
class LineDtype(GeometryDtype):
    _geometry_name = 'line'

    @classmethod
    def construct_array_type(cls, *args):
        return LineArray


class Line(GeometryList):
    _nesting_levels = 0

    @classmethod
    def construct_array_type(cls):
        return LineArray

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
        line_coords = np.array(self.data.as_py(), dtype=self.numpy_dtype)
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
        return super().from_shapely(shape)

    @property
    def length(self):
        return compute_line_length(self.buffer_values, self.buffer_inner_offsets)

    @property
    def area(self):
        return 0.0

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        result = np.zeros(1, dtype=np.bool_)
        offsets = self.buffer_outer_offsets
        lines_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, offsets[:-1], offsets[1:], result
        )
        return result[0]


class LineArray(GeometryListArray):
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
        return super().from_geopandas(ga)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        _geometry_map_nested1(
            compute_line_length,
            result,
            self.buffer_values,
            self.buffer_offsets,
            self.isna(),
        )
        return result

    @property
    def area(self):
        return np.zeros(len(self), dtype=np.float64)

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        offsets = self.buffer_outer_offsets
        start_offsets0 = offsets[:-1]
        stop_offsets0 = offsets[1:]
        if inds is not None:
            start_offsets0 = start_offsets0[inds]
            stop_offsets0 = stop_offsets0[inds]

        result = np.zeros(len(stop_offsets0), dtype=np.bool_)
        lines_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets0, stop_offsets0, result
        )
        return result


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
