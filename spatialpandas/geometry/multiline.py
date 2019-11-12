from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry1
)
import numpy as np
from spatialpandas.geometry._algorithms import (
    compute_line_length, geometry_map_nested2
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class MultiLineDtype(GeometryDtype):
    _geometry_name = 'multiline'
    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiLineArray


class MultiLine(Geometry1):
    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.MultiLineString):
            shape = list(shape)
            line_parts = []
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
        line_arrays = [line_coords.reshape(len(line_coords) // 2, 2)
                       for line_coords in np.asarray(self.data)]
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
        return super(MultiLine, cls).from_shapely(shape)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return 0.0


class MultiLineArray(GeometryArray):
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
        return super(MultiLineArray, cls).from_geopandas(ga)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        for c, result_offset in enumerate(self.offsets):
            geometry_map_nested2(
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
