from functools import total_ordering
from numbers import Integral
from typing import Iterable

import pyarrow as pa
import numpy as np
import re

from pandas.api.types import is_array_like
from pandas.api.extensions import (
    ExtensionArray, ExtensionDtype, register_extension_dtype
)

from spatialpandas.geometry._algorithms import (
    bounds_interleaved, bounds_interleaved_1d, _lexographic_lt,
    extract_isnull_bytemap)

try:
    import shapely.geometry as sg
except ImportError:
    sg = None


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


def _unwrap_geometry(a, element_dtype):
    if np.isscalar(a) and np.isnan(a):
        # replace top-level nana with None
        return None
    elif isinstance(a, Geometry):
        return np.asarray(a.data)
    elif sg and isinstance(a, sg.base.BaseGeometry):
        return element_dtype._shapely_to_coordinates(a)
    else:
        return a


class _ArrowBufferMixin(object):
    @property
    def buffer_values(self):
        buffers = self.data.buffers()
        return np.asarray(buffers[-1]).view(self.numpy_dtype)

    @property
    def buffer_offsets(self):
        buffers = self.data.buffers()
        if len(buffers) < 2:
            return (np.array([0]),)

        # Slice first offsets array to match any current extension array slice
        # All other buffers remain unchanged
        start = self.data.offset
        stop = start + len(self.data) + 1
        offsets1 = np.asarray(buffers[1]).view(np.uint32)[start:stop]

        remaining_offsets = tuple(
            np.asarray(buffers[i]).view(np.uint32)
            for i in range(3, len(buffers) - 1, 2)
        )

        return (offsets1,) + remaining_offsets

    @property
    def flat_values(self):
        # Compute valid start/stop index into buffer values array.
        buffer_offsets = self.buffer_offsets
        start = buffer_offsets[0][0]
        stop = buffer_offsets[0][-1]
        for offsets in buffer_offsets[1:]:
            start = offsets[start]
            stop = offsets[stop]

        return self.buffer_values[start:stop]


@register_extension_dtype
class GeometryDtype(ExtensionDtype):
    _geometry_name = 'geometry'
    base = np.dtype('O')
    _metadata = ('_dtype',)
    na_value = np.nan

    def __init__(self, subtype):
        if isinstance(subtype, GeometryDtype):
            self.subtype = subtype.subtype
        else:
            self.subtype = np.dtype(subtype)

        # Validate the subtype is numeric
        if self.subtype.kind not in ('i', 'u', 'f'):
            raise ValueError("Received non-numeric type of kind '{}'".format(self.kind))

        # build nested arrow type
        nesting_levels = self.construct_array_type()._nesting_levels
        arrow_dtype = pa.from_numpy_dtype(self.subtype)
        for i in range(nesting_levels):
            arrow_dtype = pa.list_(arrow_dtype)

        self.arrow_dtype = arrow_dtype

    def __hash__(self):
        return hash(self.arrow_dtype)

    def __str__(self):
        return "{}[{}]".format(self._geometry_name, str(self.subtype.name))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,  str(self.subtype.name))

    @classmethod
    def _parse_subtype(cls, dtype_string):
        """
        Parse a datatype string to get the subtype

        Parameters
        ----------
        dtype_string: str
            A string like Line2d[subtype]

        Returns
        -------
        subtype: str

        Raises
        ------
        ValueError
            When the subtype cannot be extracted
        """
        # Be case insensitive
        dtype_string = dtype_string.lower()
        subtype_re = re.compile('^' + cls._geometry_name + r"\[(?P<subtype>\w+)\]$")

        match = subtype_re.match(dtype_string)
        if match:
            subtype_string = match.groupdict()['subtype']
        elif dtype_string == cls._geometry_name.lower():
            subtype_string = 'float64'
        else:
            raise ValueError("Cannot parse {dtype_string}".format(
                dtype_string=dtype_string))

        return subtype_string

    @classmethod
    def construct_array_type(cls, *args):
        return GeometryArray

    @classmethod
    def construct_from_string(cls, string):
        # lowercase string
        string = string.lower()

        msg = "Cannot construct a '%s' from '{}'" % cls.__name__
        if string.startswith(cls._geometry_name.lower()):
            # Extract subtype
            try:
                subtype_string = cls._parse_subtype(string)
                return cls(subtype_string)
            except Exception:
                raise TypeError(msg.format(string))
        else:
            raise TypeError(msg.format(string))

    def __eq__(self, other):
        """Check whether 'other' is equal to self.
        By default, 'other' is considered equal if
        * it's a string matching 'self.name', or
        * it's an instance of this type.
        Parameters
        ----------
        other : Any
        Returns
        -------
        bool
        """
        if isinstance(other, type(self)):
            return self.subtype == other.subtype
        elif isinstance(other, str):
            return str(self) == other
        else:
            return False

    @property
    def type(self):
        # type: () -> type
        """The scalar type for the array, e.g. ``int``.
        It's expected ``ExtensionArray[item]`` returns an instance
        of ``ExtensionDtype.type`` for scalar ``item``.
        """
        return Geometry

    @property
    def name(self):
        # type: () -> str
        """A string identifying the data type.
        Will be used for display in, e.g. ``Series.dtype``
        """
        return str(self)


@total_ordering
class Geometry(_ArrowBufferMixin):
    _nesting_levels = 0

    def __init__(self, data):
        if isinstance(data, pa.ListArray):
            # Use arrow ListArray as is
            self.data = data
        else:
            self.data = pa.array(data)
        if len(self.data) > 0:
            _validate_nested_arrow_type(self._nesting_levels, self.data.type)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data.to_pylist())

    def __hash__(self):
        return hash(np.asarray(self.data).tobytes())

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return np.array_equal(np.asarray(self.data), np.asarray(other.data))

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return _lexographic_lt(np.asarray(self.data), np.asarray(other.data))

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        raise NotImplementedError()

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas geometry object from a shapely shape

        Args:
            shape: A shapely shape

        Returns:
            spatialpandas geometry object with type of the calling class
        """
        shape_parts = cls._shapely_to_coordinates(shape)
        return cls(shape_parts)

    @property
    def numpy_dtype(self):
        return self.data.type.to_pandas_dtype()().dtype


class Geometry0(Geometry):
    _nesting_levels = 0

    @property
    def numpy_dtype(self):
        return self.data.type.to_pandas_dtype()().dtype

    @property
    def _values(self):
        return np.asarray(self.data)

    @property
    def _value_offsets(self):
        return np.array([0, len(self.data)])


class Geometry1(Geometry):
    _nesting_levels = 1

    @property
    def numpy_dtype(self):
        if isinstance(self.data, pa.NullArray):
            return None
        else:
            return self.data.type.value_type.to_pandas_dtype()().dtype

    @property
    def _value_offsets(self):
        buffers = self.data.buffers()
        if len(buffers) <= 1:
            # Treat NullArray as empty double array so numba can handle it
            return np.array([0], dtype=np.uint32)
        else:
            offsets0 = np.asarray(buffers[1]).view(np.uint32)

            start = self.data.offset
            stop = start + len(self.data) + 1
            return offsets0[start:stop]

    @property
    def _values(self):
        if isinstance(self.data, pa.NullArray):
            # Treat NullArray as empty double array so numba can handle it
            return np.array([], dtype=np.float64)
        else:
            return self.data.flatten().to_numpy()


class Geometry2(Geometry):
    _nesting_levels = 2

    @property
    def numpy_dtype(self):
        if isinstance(self.data, pa.NullArray):
            return None
        else:
            return self.data.type.value_type.value_type.to_pandas_dtype()().dtype

    @property
    def _value_offsets(self):
        buffers = self.data.buffers()
        if len(buffers) <= 1:
            # Treat NullArray as empty double array so numba can handle it
            return np.array([0], dtype=np.uint32)
        else:
            offsets0 = np.asarray(buffers[1]).view(np.uint32)
            offsets1 = np.asarray(buffers[3]).view(np.uint32)

            start = offsets0[self.data.offset]
            stop = offsets0[self.data.offset + len(self.data)]
            return offsets1[start:stop + 1]

    @property
    def _values(self):
        if isinstance(self.data, pa.NullArray):
            # Treat NullArray as empty double array so numba can handle it
            return np.array([], dtype=np.float64)
        else:
            return self.data.flatten().flatten().to_numpy()



class GeometryArray(ExtensionArray, _ArrowBufferMixin):
    _can_hold_na = True
    _element_type = Geometry
    _nesting_levels = 1

    # Import / export methods
    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas geometry array from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries to import

        Returns:
            spatialpandas geometry array with type of the calling class
        """
        if cls is GeometryArray:
            raise ValueError(
                "from_geopandas must be called on a subclass of GeometryArray"
            )
        return cls([cls._element_type._shapely_to_coordinates(shape) for shape in ga])

    def to_geopandas(self):
        """
        Convert a spatialpandas geometry array into a geopandas GeometryArray

        Returns:
            geopandas GeometryArray
        """
        from geopandas.array import from_shapely
        return from_shapely([el.to_shapely() for el in self])

    def __arrow_array__(self, type=None):
        return self.data

    @classmethod
    def __from_arrow__(cls, data):
        return cls(data)

    # Constructor
    def __init__(self, array, dtype=None, copy=None):
        # Choose default dtype for empty arrays
        try:
            if len(array) == 0 and dtype is None:
                dtype = 'float64'
        except:
            # len failed
            pass

        if isinstance(dtype, GeometryDtype):
            # Use arrow type as-is
            arrow_dtype = dtype.arrow_dtype
        elif dtype is not None and dtype != np.dtype('object'):
            # Scalar element dtype
            arrow_dtype = pa.from_numpy_dtype(dtype)

            # Wrap dtype with appropriate number of nesting levels
            for i in range(self._nesting_levels):
                arrow_dtype = pa.list_(arrow_dtype)
        else:
            # Let arrow infer type
            arrow_dtype = None

        # Unwrap Geometry elements to numpy arrays
        if is_array_like(array) or isinstance(array, list):
            array = [_unwrap_geometry(el, self._element_type) for el in array]
            self.data = pa.array(array, type=arrow_dtype)
        elif isinstance(array, pa.Array):
            self.data = array
        elif isinstance(array, pa.ChunkedArray):
            self.data = pa.concat_arrays(array)
        else:
            raise ValueError(
                "Unsupported type passed for {}: {}".format(
                    self.__class__.__name__, type(array)
                )
            )

        self.offsets = np.array([0])

        # Check that inferred type has the right number of nested levels
        pyarrow_element_type = _validate_nested_arrow_type(
            self._nesting_levels, self.data.type
        )

        self._pyarrow_element_type = pyarrow_element_type
        self._numpy_element_type = pyarrow_element_type.to_pandas_dtype()
        self._dtype = self._dtype_class(self._numpy_element_type)

        # Validate input data is compatible
        offsets = self.buffer_offsets

        # Validate even number of inner elements per polygon
        if any((offsets[-1] % 2) > 0):
            raise ValueError("""
Geometry objects are represented by interleaved x and y coordinates, so they must have
an even number of elements. Received specification with an odd number of elements.""")

    @property
    def _dtype_class(self):
        return GeometryDtype

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        size = 0
        for buf in self.data.buffers():
            if buf is not None:
                size += buf.size
        return size

    nbytes.__doc__ = ExtensionArray.nbytes.__doc__

    def astype(self, dtype, copy=True):
        if self.dtype == dtype:
            return self

        if dtype == np.dtype('object'):
            return np.array(self, dtype='object')

        if isinstance(dtype, GeometryDtype):
            dtype = dtype.arrow_dtype.to_pandas_dtype()
        elif isinstance(dtype, pa.DataType):
            dtype = dtype.to_pandas_dtype()
        else:
            dtype = np.dtype(dtype)

        return GeometryArray(np.asarray(self).astype(dtype), dtype=dtype)

    astype.__doc__ = ExtensionArray.astype.__doc__

    def isna(self):
        return extract_isnull_bytemap(self.data)

    isna.__doc__ = ExtensionArray.isna.__doc__

    def copy(self):
        return type(self)(self.data)

    copy.__doc__ = ExtensionArray.copy.__doc__

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        err_msg = ("Only integers, slices and integer or boolean"
                   "arrays are valid indices.")
        if isinstance(item, Integral):
            item = int(item)
            if item < -len(self) or item >= len(self):
                raise IndexError("{item} is out of bounds".format(item=item))
            else:
                # Convert negative item index
                if item < 0:
                    item += len(self)

                value = self.data[item].as_py()
                if value is not None:
                    return self._element_type(value)
                else:
                    return None
        elif isinstance(item, slice):
            data = []
            selected_indices = np.arange(len(self))[item]

            for selected_index in selected_indices:
                data.append(self[selected_index])

            return self.__class__(data, dtype=self.dtype)
        elif isinstance(item, Iterable):
            item = np.asarray(item)
            if item.dtype == 'bool':
                data = []

                for i, m in enumerate(item):
                    if m:
                        data.append(self[i])

                return self.__class__(data, dtype=self.dtype)

            elif item.dtype.kind in ('i', 'u'):
                return self.take(item, allow_fill=False)
            else:
                raise IndexError(err_msg)
        else:
            raise IndexError(err_msg)

    @property
    def numpy_dtype(self):
        return self._numpy_element_type().dtype

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=None):
        if isinstance(scalars, GeometryArray):
            return scalars

        return cls(scalars, dtype=dtype)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.core.algorithms import take

        data = self.astype(object)
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.
        result = take(data, indices, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype.subtype)

    take.__doc__ = ExtensionArray.take.__doc__

    def _values_for_factorize(self):
        return np.array(self, dtype='object'), None

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def _values_for_argsort(self):
        return np.array(list(self), dtype='object')

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(
            pa.concat_arrays(
                [ea.data for ea in to_concat]
            ),
            dtype=to_concat[0].dtype
        )

    def fillna(self, value=None, method=None, limit=None):
        from pandas.api.types import is_array_like
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import pad_1d, backfill_1d

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if is_array_like(value):
            if len(value) != len(self):
                raise ValueError(
                    "Length of 'value' does not match. Got ({}) "
                    " expected {}".format(len(value), len(self))
                )
            value = value[mask]

        if mask.any():
            if method is not None:
                func = pad_1d if method == "pad" else backfill_1d
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, self._dtype)
            else:
                # fill with value
                new_values = np.asarray(self)
                if isinstance(value, Geometry):
                    value = [value]
                new_values[mask] = value
                new_values = self.__class__(new_values)
        else:
            new_values = self.copy()
        return new_values

    fillna.__doc__ = ExtensionArray.fillna.__doc__

    # Base geometry methods
    @property
    def bounds(self):
        return bounds_interleaved(self.flat_values)

    @property
    def bounds_x(self):
        return bounds_interleaved_1d(self.flat_values, 0)

    @property
    def bounds_y(self):
        return bounds_interleaved_1d(self.flat_values, 1)
