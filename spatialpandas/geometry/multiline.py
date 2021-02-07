import numpy as np
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry._algorithms.intersection import (lines_intersect_bounds,
                                                 multilines_intersect_bounds)
from ..geometry._algorithms.measures import compute_line_length
from ..geometry.base import GeometryDtype
from ..geometry.baselist import (GeometryList, GeometryListArray,
                                 _geometry_map_nested2)


@register_extension_dtype
class MultiLineDtype(GeometryDtype):
    _geometry_name = 'multiline'
    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiLineArray


class MultiLine(GeometryList):
    _nesting_levels = 1

    @classmethod
    def construct_array_type(cls):
        return MultiLineArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.MultiLineString):
            shape = list(shape)
            line_parts = []
            if len(shape) == 0:
                # Add single empty line so we have the right number of nested levels
                line_parts.append([])
            else:
                for line in shape:
                    line_parts.append(np.asarray(line.ctypes))
            return line_parts
        elif isinstance(shape, (sg.LineString, sg.LinearRing)):
            return [np.asarray(shape.ctypes)]
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of MultiLineString
""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely MultiLineString shape
        """
        import shapely.geometry as sg

        def to_numpy_2d(v):
            a = np.array(v.as_py(), dtype=self.numpy_dtype)
            return a.reshape(len(v) // 2, 2)

        line_arrays = [to_numpy_2d(line_coords) for line_coords in self.data]
        lines = [sg.LineString(line_array) for line_array in line_arrays]
        return sg.MultiLineString(lines=lines)

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas MultiLine object from a shapely shape

        Args:
            shape: A shapely MultiLineString, LineString, or LinearRing shape

        Returns:
            spatialpandas MultiLine
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
        offsets = self.buffer_outer_offsets
        start_offsets = offsets[:-1]
        stop_offstes = offsets[1:]
        result = np.zeros(len(start_offsets), dtype=np.bool_)
        lines_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets, stop_offstes, result
        )
        return result.any()


class MultiLineArray(GeometryListArray):
    _element_type = MultiLine
    _nesting_levels = 2

    @property
    def _dtype_class(self):
        return MultiLineDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas MultiLineArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiLineString,
            LineString, or LinearRing shapes.

        Returns:
            MultiLineArray
        """
        return super().from_geopandas(ga)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        _geometry_map_nested2(
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
        offsets0, offsets1 = self.buffer_offsets
        start_offsets0 = offsets0[:-1]
        stop_offsets0 = offsets0[1:]
        if inds is not None:
            start_offsets0 = start_offsets0[inds]
            stop_offsets0 = stop_offsets0[inds]

        result = np.zeros(len(start_offsets0), dtype=np.bool_)
        multilines_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets0, stop_offsets0, offsets1, result
        )
        return result


def _multi_line_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return MultiLineArray([
        [[1, 0, 1, 1], [1, 2, 0, 0]],
        [[3, 3, 4, 4]]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(MultiLineDtype)(_multi_line_array_non_empty)
