from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry._algorithms import compute_line_length, \
    geometry_map_nested1
from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry0
)
import numpy as np
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class Line2dDtype(GeometryDtype):
    _geometry_name = 'line2d'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return Line2dArray


class Line2d(Geometry0):
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
        import shapely.geometry as sg
        line_coords = self.data.to_numpy()
        return sg.LineString(line_coords.reshape(len(line_coords) // 2, 2))

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return 0.0


class Line2dArray(GeometryArray):
    _element_type = Line2d
    _nesting_levels = 1

    @property
    def _dtype_class(self):
        return Line2dDtype

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


def line_array_non_empty(dtype):
    return Line2dArray([
        [1, 0, 1, 1],
        [1, 2, 0, 0]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(Line2dDtype)(line_array_non_empty)
