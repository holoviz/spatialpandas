from functools import total_ordering

import numpy as np
import pyarrow as pa
from numba import jit, prange

from ..geometry.base import Geometry, GeometryArray
from ._algorithms.bounds import (bounds_interleaved, total_bounds_interleaved,
                                 total_bounds_interleaved_1d)


def _validate_nested_arrow_type(nesting_levels, pyarrow_type):
    if pyarrow_type == pa.null():
        return pa.null()

    pyarrow_element_type = pyarrow_type
    for i in range(nesting_levels):
        if not isinstance(pyarrow_element_type, pa.ListType):
            raise ValueError(
                "Expected input data to have {} nested layer(s)".format(
                    nesting_levels)
            )
        pyarrow_element_type = pyarrow_element_type.value_type
    pyarrow_element_type = pyarrow_element_type
    numpy_element_dtype = pyarrow_element_type.to_pandas_dtype()
    if (numpy_element_dtype() is None
            or numpy_element_dtype().dtype.kind not in ('i', 'u', 'f')):
        raise ValueError(
            "Invalid nested element type {}, expected numeric type".format(
                pyarrow_element_type
            ))
    return pyarrow_element_type


class _ListArrayBufferMixin:
    """
    Mixin of buffer utilities for classes that store a pyarrow ListArray as their
    listarray property. The numpy data type of the inner ListArray elements must be
    stored as the numpy_dtype property
    """
    @property
    def buffer_values(self):
        value_buffer = self.listarray.buffers()[-1]
        if value_buffer is None:
            return np.array([], dtype=self.numpy_dtype)
        else:
            return np.asarray(value_buffer).view(self.numpy_dtype)

    @property
    def buffer_offsets(self):
        """
        Tuple of offsets arrays, one for each tested level.
        """
        buffers = self.listarray.buffers()
        if len(buffers) < 2:
            return (np.array([0]),)
        elif len(buffers) < 3:
            # offset values that include everything
            return (np.array([0, len(self.listarray)]),)

        # Slice first offsets array to match any current extension array slice
        # All other buffers remain unchanged
        start = self.listarray.offset
        stop = start + len(self.listarray) + 1
        offsets1 = np.asarray(buffers[1]).view(np.uint32)[start:stop]

        remaining_offsets = tuple(
            np.asarray(buffers[i]).view(np.uint32)
            for i in range(3, len(buffers) - 1, 2)
        )

        return (offsets1,) + remaining_offsets

    @property
    def flat_values(self):
        """
        Flat array of the valid values. This differs from buffer_values if the pyarrow
        ListArray backing this object is a slice. buffer_values will contain all
        values from the original (pre-sliced) object whereas flat_values will contain
        only the sliced values.
        """
        # Compute valid start/stop index into buffer values array.
        buffer_offsets = self.buffer_offsets
        start = buffer_offsets[0][0]
        stop = buffer_offsets[0][-1]
        for offsets in buffer_offsets[1:]:
            start = offsets[start]
            stop = offsets[stop]

        return self.buffer_values[start:stop]

    @property
    def buffer_outer_offsets(self):
        """
        Array of the offsets into buffer_values that separate the outermost nested
        structure of geometry object(s), regardless of the number of nesting levels.
        """
        buffer_offsets = self.buffer_offsets
        flat_offsets = buffer_offsets[0]
        for offsets in buffer_offsets[1:]:
            flat_offsets = offsets[flat_offsets]

        return flat_offsets

    @property
    def buffer_inner_offsets(self):
        """
        Array of the offsets into buffer_values that separate the innermost nested
        structure of geometry object(s), regardless of the number of nesting levels.
        """
        buffer_offsets = self.buffer_offsets
        start = buffer_offsets[0][0]
        stop = buffer_offsets[0][-1]
        for offsets in buffer_offsets[1:-1]:
            start = offsets[start]
            stop = offsets[stop]
        return buffer_offsets[-1][start:stop + 1]


@total_ordering
class GeometryList(Geometry, _ListArrayBufferMixin):
    """
    Base class for elements of GeometryListArray subclasses
    """
    _nesting_levels = 0

    @staticmethod
    def _pa_element_value_type(data):
        """
        Get value type of pyarrow ListArray element for different versions of pyarrow
        """
        try:
            # Try pyarrow 1.0 API
            return data.values.type
        except AttributeError:
            # Try pre 1.0 API
            return data.value_type

    @staticmethod
    def _pa_element_values(data):
        """
        Get values of nested pyarrow ListArray element for different versions of pyarrow
        """
        try:
            # Try pyarrow 1.0 API
            return data.values
        except AttributeError:
            # Try pre 1.0 API
            return pa.array(data.as_py(), data.value_type)

    def __init__(self, data, dtype=None):
        super().__init__(data)
        if len(self.data) > 0:
            value_type = GeometryList._pa_element_value_type(self.data)
            _validate_nested_arrow_type(self._nesting_levels, value_type)

        # create listarray for _ListArrayBufferMixin
        self.listarray = GeometryList._pa_element_values(self.data)

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return _lexographic_lt(np.asarray(self.listarray), np.asarray(other.listarray))

    def __len__(self):
        return len(self.buffer_outer_offsets - 1)

    @classmethod
    def construct_array_type(cls):
        return GeometryListArray

    @property
    def numpy_dtype(self):
        if isinstance(self.listarray, pa.NullArray):
            return None
        else:
            typ = GeometryList._pa_element_value_type(self.data)
            for _ in range(self._nesting_levels):
                typ = GeometryList._pa_element_value_type(typ)
            return np.dtype(typ.to_pandas_dtype())


class GeometryListArray(GeometryArray, _ListArrayBufferMixin):
    """
    Base class for geometry arrays that are backed by a pyarrow ListArray.
    """
    _element_type = GeometryList
    _nesting_levels = 1

    @classmethod
    def _arrow_type_from_numpy_element_dtype(cls, dtype):
        # Scalar element dtype
        arrow_dtype = pa.from_numpy_dtype(dtype)

        # Wrap dtype with appropriate number of nesting levels
        for i in range(cls._nesting_levels):
            arrow_dtype = pa.list_(arrow_dtype)

        return arrow_dtype

    def _numpy_element_dtype_from_arrow_type(self, pyarrow_type):
        if pyarrow_type == pa.null():
            return pa.null()

        pyarrow_element_type = pyarrow_type
        for i in range(self._nesting_levels):
            pyarrow_element_type = pyarrow_element_type.value_type

        return pyarrow_element_type.to_pandas_dtype()

    # Constructor
    def __init__(self, array, dtype=None):
        super().__init__(array, dtype)

        # Set listarray property for _ListArrayBufferMixin
        self.listarray = self.data

        # Check that inferred type has the right number of nested levels
        _validate_nested_arrow_type(
            self._nesting_levels, self.data.type
        )

        # Validate input data is compatible
        offsets = self.buffer_offsets

        # Validate even number of inner elements per polygon
        if any((offsets[-1] % 2) > 0):
            raise ValueError("""
GeometryList objects are represented by interleaved x and y coordinates, so they must have
an even number of elements. Received specification with an odd number of elements.""")

    # Base geometry methods
    @property
    def total_bounds(self):
        return total_bounds_interleaved(self.flat_values)

    @property
    def total_bounds_x(self):
        return total_bounds_interleaved_1d(self.flat_values, 0)

    @property
    def total_bounds_y(self):
        return total_bounds_interleaved_1d(self.flat_values, 1)

    @property
    def bounds(self):
        return bounds_interleaved(self.buffer_values, self.buffer_outer_offsets)


@jit(nopython=True, nogil=True)
def _lexographic_lt0(a1, a2):
    """
    Compare two 1D numpy arrays lexographically
    Parameters
    ----------
    a1: ndarray
        1D numpy array
    a2: ndarray
        1D numpy array

    Returns
    -------
    comparison:
        True if a1 < a2, False otherwise
    """
    for e1, e2 in zip(a1, a2):
        if e1 < e2:
            return True
        elif e1 > e2:
            return False
    return len(a1) < len(a2)


def _lexographic_lt(a1, a2):
    if a1.dtype != np.dtype(object) and a1.dtype != np.dtype(object):
        # a1 and a2 primitive
        return _lexographic_lt0(a1, a2)
    elif a1.dtype == np.dtype(object) and a1.dtype == np.dtype(object):
        # a1 and a2 object, process recursively
        for e1, e2 in zip(a1, a2):
            if _lexographic_lt(e1, e2):
                return True
            elif _lexographic_lt(e2, e1):
                return False
        return len(a1) < len(a2)
    elif a1.dtype != np.dtype(object):
        # a2 is object array, a1 primitive
        return True
    else:
        # a1 is object array, a2 primitive
        return False


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested1(
        fn, result, values, value_offsets, missing
):
    assert len(value_offsets) == 1
    value_offsets0 = value_offsets[0]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i] = fn(values, value_offsets0[i:i + 2])


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested2(
        fn, result, values, value_offsets, missing
):
    assert len(value_offsets) == 2
    value_offsets0 = value_offsets[0]
    value_offsets1 = value_offsets[1]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = value_offsets0[i]
            stop = value_offsets0[i + 1]
            result[i] = fn(values, value_offsets1[start:stop + 1])


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested3(
        fn, result, values, value_offsets, missing
):
    assert len(value_offsets) == 3
    value_offsets0 = value_offsets[0]
    value_offsets1 = value_offsets[1]
    value_offsets2 = value_offsets[2]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = value_offsets1[value_offsets0[i]]
            stop = value_offsets1[value_offsets0[i + 1]]
            result[i] = fn(values, value_offsets2[start:stop + 1])
