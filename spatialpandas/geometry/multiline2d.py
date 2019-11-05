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
class MultiLine2dDtype(GeometryDtype):
    _geometry_name = 'multiline2d'
    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiLine2dArray


class MultiLine2d(Geometry1):
    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.MultiLineString):
            shape = list(shape)
            line_parts = []
            for line in shape:
                line_parts.append(np.asarray(line.ctypes))
            return line_parts
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of MultiLineString
""".format(typ=type(shape).__name__))

    def to_shapely(self):
        import shapely.geometry as sg
        line_arrays = [line_coords.reshape(len(line_coords) // 2, 2)
                       for line_coords in np.asarray(self.data)]
        lines = [sg.LineString(line_array) for line_array in line_arrays]
        return sg.MultiLineString(lines=lines)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return 0.0


class MultiLine2dArray(GeometryArray):
    _element_type = MultiLine2d
    _nesting_levels = 2

    @property
    def _dtype_class(self):
        return MultiLine2dDtype

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


def multi_line_array_non_empty(dtype):
    return MultiLine2dArray([
        [[1, 0, 1, 1], [1, 2, 0, 0]],
        [[3, 3, 4, 4]]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(MultiLine2dDtype)(multi_line_array_non_empty)
