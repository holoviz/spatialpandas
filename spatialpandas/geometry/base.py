from __future__ import absolute_import
import re
from functools import total_ordering
import numpy as np
from numba import prange, jit
from pandas.core.dtypes.dtypes import register_extension_dtype

from ._ragged import _RaggedElement, RaggedDtype, RaggedArray


ngjit = jit(nopython=True, nogil=True)


@total_ordering
class Geometry(_RaggedElement):
    def __repr__(self):
        data = [(x, y) for x, y in zip(self.xs, self.ys)]
        return "{typ}({data})".format(typ=self.__class__.__name__, data=data)

    @classmethod
    def _shapely_to_array_parts(cls, shape):
        raise NotImplementedError()

    @classmethod
    def from_shapely(cls, shape):
        shape_parts = cls._shapely_to_array_parts(shape)
        return cls(np.concatenate(shape_parts))

    @property
    def xs(self):
        return self.array[0::2]

    @property
    def ys(self):
        return self.array[1::2]

    @property
    def bounds(self):
        return bounds_interleaved(self.array)

    @property
    def bounds_x(self):
        return bounds_interleaved_1d(self.array, 0)

    @property
    def bounds_y(self):
        return bounds_interleaved_1d(self.array, 1)

    @property
    def length(self):
        raise NotImplementedError()

    @property
    def area(self):
        raise NotImplementedError()


@register_extension_dtype
class GeometryDtype(RaggedDtype):
    _type_name = "Geometry"
    _subtype_re = re.compile(r"^geom\[(?P<subtype>\w+)\]$")

    @classmethod
    def construct_array_type(cls):
        return GeometryArray


class GeometryArray(RaggedArray):
    _element_type = Geometry

    def __init__(self, *args, **kwargs):
        super(GeometryArray, self).__init__(*args, **kwargs)
        # Validate that there are an even number of elements in each Geometry element
        if (any(self.start_indices % 2) or
                len(self) and (len(self.flat_array) - self.start_indices[-1]) % 2 > 0):
            raise ValueError("There must be an even number of elements in each row")

    @property
    def _dtype_class(self):
        return GeometryDtype

    @property
    def xs(self):
        start_indices = self.start_indices // 2
        flat_array = self.flat_array[0::2]
        return RaggedArray({"start_indices": start_indices, "flat_array": flat_array})

    @property
    def ys(self):
        start_indices = self.start_indices // 2
        flat_array = self.flat_array[1::2]
        return RaggedArray({"start_indices": start_indices, "flat_array": flat_array})

    def to_geopandas(self):
        from geopandas.array import from_shapely
        return from_shapely([el.to_shapely() for el in self])

    @classmethod
    def from_geopandas(cls, ga):
        shape_parts = [
            cls._element_type._shapely_to_array_parts(shape) for shape in ga
        ]
        shape_lengths = [
            sum([len(part) for part in parts])
            for parts in shape_parts
        ]
        flat_array = np.concatenate(
            [part for parts in shape_parts for part in parts]
        )
        start_indices = np.concatenate(
            [[0], shape_lengths[:-1]]
        ).astype('uint').cumsum()
        return cls({
            'start_indices': start_indices, 'flat_array': flat_array
        })

    @property
    def bounds(self):
        return bounds_interleaved(self.flat_array)

    @property
    def bounds_x(self):
        return bounds_interleaved_1d(self.flat_array, 0)

    @property
    def bounds_y(self):
        return bounds_interleaved_1d(self.flat_array, 1)

    @property
    def length(self):
        raise NotImplementedError()

    @property
    def area(self):
        raise NotImplementedError()


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map(start_indices, flat_array, result, fn):
    n = len(start_indices)
    for i in prange(n):
        start = start_indices[i]
        stop = start_indices[i + 1] if i < n - 1 else len(flat_array)
        result[i] = fn(flat_array[start:stop])


@ngjit
def bounds_interleaved(values):
    """
    compute bounds
    """
    xmin = np.inf
    ymin = np.inf
    xmax = -np.inf
    ymax = -np.inf

    for i in range(0, len(values), 2):
        x = values[i]
        if np.isfinite(x):
            xmin = min(xmin, x)
            xmax = max(xmax, x)

        y = values[i + 1]
        if np.isfinite(y):
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    return (xmin, ymin, xmax, ymax)


@ngjit
def bounds_interleaved_1d(values, offset):
    """
    compute bounds
    """
    vmin = np.inf
    vmax = -np.inf

    for i in range(0, len(values), 2):
        v = values[i + offset]
        if np.isfinite(v):
            vmin = min(vmin, v)
            vmax = max(vmax, v)

    return (vmin, vmax)
