from functools import total_ordering

import numpy as np
import pyarrow as pa

from ..geometry.base import Geometry, GeometryArray, GeometryDtype
from ..geometry.baselist import _lexographic_lt
from ._algorithms.bounds import (bounds_interleaved, total_bounds_interleaved,
                                 total_bounds_interleaved_1d)


@total_ordering
class GeometryFixed(Geometry):
    """
    Base class for elements of GeometryFixedArray subclasses
    """
    def __init__(self, data, dtype=None):
        if isinstance(data, np.ndarray):
            # Convert numpy array to bytes
            dtype = data.dtype
            data = data.tobytes()

        super().__init__(data)
        self.numpy_dtype = np.dtype(dtype)
        self.pyarrow_type = pa.from_numpy_dtype(dtype)

    def __lt__(self, other):
        return _lexographic_lt(self.flat_values, other.flat_values)

    def __len__(self):
        return 8 * len(self.data.as_py()) // self.pyarrow_type.bit_width

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.flat_values.tolist())

    @property
    def flat_values(self):
        return pa.Array.from_buffers(
            self.pyarrow_type,
            len(self),
            buffers=[None, pa.py_buffer(self.data.as_py())]
        ).to_numpy()

    @classmethod
    def construct_array_type(cls):
        return GeometryFixedArray


class GeometryFixedArray(GeometryArray):
    """
    Base class for geometry arrays that are backed by a pyarrow fixed width binary type
    """
    _element_type = GeometryFixed
    _element_len = 2

    @classmethod
    def _arrow_type_from_numpy_element_dtype(cls, dtype):
        # Scalar element dtype
        arrow_dtype = pa.from_numpy_dtype(dtype)
        return pa.binary(arrow_dtype.bit_width // 8)

    def _numpy_element_dtype_from_arrow_type(self, pyarrow_type):
        return self._numpy_dtype

    # Constructor
    def __init__(self, array, dtype=None):

        def invalid_array():
            err_msg = (
                "Invalid array with type {typ}\n"
                "A {cls} may be constructed from:\n"
                "    - A 1-d array with length divisible by {n} of interleaved\n"
                "      x y coordinates\n"
                "    - A tuple of {n} 1-d arrays\n"
                "    - A pyarrow.FixedSizeBinaryArray. In this case the dtype\n"
                "      argument must also be specified"
            ).format(
                typ=type(array), cls=self.__class__.__name__, n=self._element_len,
            )
            raise ValueError(err_msg)

        if isinstance(dtype, GeometryDtype):
            dtype = dtype.subtype

        numpy_dtype = None
        pa_type = None
        if isinstance(array, (pa.Array, pa.ChunkedArray)):
            if dtype is None:
                invalid_array()
            numpy_dtype = np.dtype(dtype)
        elif isinstance(array, tuple):
            if len(array) == self._element_len:
                array = [np.asarray(array[i]) for i in range(len(array))]
                if dtype:
                    array = [array[i].astype(dtype) for i in range(len(array))]

                # Capture numpy dtype
                numpy_dtype = array[0].dtype

                # Create buffer and FixedSizeBinaryArray
                pa_type = pa.binary(
                    self._element_len * pa.from_numpy_dtype(numpy_dtype).bit_width // 8
                )
                buffers = [None, pa.py_buffer(np.stack(array, axis=1).tobytes())]
                array = pa.Array.from_buffers(
                    pa_type, len(array[0]), buffers=buffers
                )
            else:
                invalid_array()
        else:
            array = np.asarray(array)
            if array.dtype.kind == 'O':
                if array.ndim != 1:
                    invalid_array()

                # Try to infer dtype
                if dtype is None:
                    for i in range(len(array)):
                        el = array[i]
                        if el is None:
                            continue
                        if isinstance(el, GeometryFixed):
                            numpy_dtype = el.numpy_dtype
                        else:
                            el_array = np.asarray(el)
                            numpy_dtype = el_array.dtype
                        break
                    if numpy_dtype is None:
                        invalid_array()
                else:
                    numpy_dtype = dtype

                # Explicitly set the pyarrow binary type
                pa_type = pa.binary(
                    self._element_len * pa.from_numpy_dtype(numpy_dtype).bit_width // 8
                )

                # Convert individual rows to bytes
                array = array.copy()
                for i in range(len(array)):
                    el = array[i]
                    if el is None:
                        continue
                    if isinstance(el, bytes):
                        # Nothing to do
                        pass
                    elif isinstance(el, GeometryFixed):
                        array[i] = el.flat_values.tobytes()
                    else:
                        array[i] = np.asarray(el, dtype=numpy_dtype).tobytes()
            else:
                if dtype:
                    array = array.astype(dtype)

                # Capture numpy dtype
                numpy_dtype = array.dtype
                pa_type = pa.binary(
                    self._element_len * pa.from_numpy_dtype(numpy_dtype).bit_width // 8
                )

                if array.ndim == 2:
                    # Handle 2d array case
                    if array.shape[1] != self._element_len:
                        invalid_array()
                    # Create buffer and FixedSizeBinaryArray
                    buffers = [None, pa.py_buffer(array.tobytes())]
                    array = pa.Array.from_buffers(
                        pa_type, array.shape[0], buffers=buffers
                    )
                elif array.ndim == 1 and len(array) % self._element_len == 0:
                    buffers = [None, pa.py_buffer(array.tobytes())]
                    array = pa.Array.from_buffers(
                        pa_type, len(array) // self._element_len, buffers=buffers
                    )
                else:
                    invalid_array()

        self._numpy_dtype = numpy_dtype
        super().__init__(array, pa_type)

    # Base geometry methods
    @property
    def flat_values(self):
        if len(self.data) == 0:
            return np.array([], dtype=self.numpy_dtype)
        else:
            start = self.data.offset * self._element_len
            stop = start + len(self.data) * self._element_len
            return np.asarray(self.data.buffers()[1]).view(self.numpy_dtype)[start:stop]

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
        flat_values = self.flat_values
        if len(flat_values) == 0:
            return np.zeros((0, 4), dtype=self.numpy_dtype)
        else:
            bounds = np.full((len(self), 4), np.nan, dtype=np.float64)
            bounds[~self.isna(), :] = bounds_interleaved(
                flat_values, np.arange(0, len(flat_values) + 1, self._element_len)
            )
            return bounds
