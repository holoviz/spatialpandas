from functools import total_ordering
from numbers import Integral
from typing import Iterable
import pyarrow as pa
import numpy as np
import re

from numba import jit, prange
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.core.dtypes.inference import is_array_like

from ._algorithms.bounds import (
    total_bounds_interleaved, total_bounds_interleaved_1d, bounds_interleaved
)
from ..spatialindex import HilbertRtree
from ..utils import ngjit
from .._optional_imports import sg, gp


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
        value_buffer = self.data.buffers()[-1]
        if value_buffer is None:
            return np.array([], dtype=self.numpy_dtype)
        else:
            return np.asarray(value_buffer).view(self.numpy_dtype)

    @property
    def buffer_offsets(self):
        buffers = self.data.buffers()
        if len(buffers) < 2:
            return (np.array([0]),)
        elif len(buffers) < 3:
            # offset values that include everything
            return (np.array([0, len(self.data)]),)

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
        return hash((self.__class__, self.arrow_dtype))

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
            A string like line[subtype]

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
        return hash((self.__class__, np.asarray(self.data).tobytes()))

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return np.array_equal(np.asarray(self.data), np.asarray(other.data))

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return _lexographic_lt(np.asarray(self.data), np.asarray(other.data))

    def __len__(self):
        return len(self.buffer_outer_offsets - 1)

    @classmethod
    def construct_array_type(cls):
        return GeometryArray

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
        if isinstance(self.data, pa.NullArray):
            return None
        else:
            typ = self.data.type
            for _ in range(self._nesting_levels):
                typ = typ.value_type
            return np.dtype(typ.to_pandas_dtype())

    def intersects_bounds(self, bounds):
        raise NotImplementedError()


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
        elif isinstance(array, dict) and 'offsets' in array and 'values' in array:
            # Dict of flat values / offsets
            offsets = array['offsets']
            values = array['values']
            isna = array.get('isna', None)

            # Build list of missing bitmasks
            masks = [None] * len(offsets)
            if isna is not None:
                mask = np.packbits(~isna, bitorder='little')
                masks[0] = pa.py_buffer(mask)

            # Build scalar arrow dtype
            arrow_dtype = pa.from_numpy_dtype(values.dtype)

            # Build inner ListArray
            array = pa.ListArray.from_buffers(
                arrow_dtype, length=len(values), buffers=[None, pa.py_buffer(values)]
            )

            # Wrap nested lists
            for i in reversed(range(len(offsets))):
                offset = offsets[i]
                mask = masks[i]
                arrow_dtype = pa.list_(arrow_dtype)
                array = pa.ListArray.from_buffers(
                    arrow_dtype,
                    length=len(offset) - 1,
                    buffers=[mask, pa.py_buffer(offset)],
                    children=[array]
                )

            self.data = array
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

        # Initialize backing property for spatial index
        self._sindex = None

    def __eq__(self, other):
        if type(other) is type(self):
            if len(other) != len(self):
                raise ValueError("""
Cannot check equality of {typ} instances of unequal length
    len(ra1) == {len_a1}
    len(ra2) == {len_a2}""".format(
                    typ=type(self).__name__,
                    len_a1=len(self),
                    len_a2=len(other)))
            result = np.zeros(len(self), dtype=np.bool_)
            self_offsets = self.buffer_offsets
            other_offsets = other.buffer_offsets
            self_values = self.buffer_values
            other_values = other.buffer_values
            for i in range(len(self)):
                self_start = other_start = i
                self_stop = other_stop = i + 1

                # Recurse through nested levels and look for a length mismatch
                length_mismatch = False
                for j in range(len(self_offsets)):
                    self_start = self_offsets[j][self_start]
                    self_stop = self_offsets[j][self_stop]
                    other_start = other_offsets[j][other_start]
                    other_stop = other_offsets[j][other_stop]

                    if self_stop - self_start != other_stop - other_start:
                        length_mismatch = True
                        break

                if length_mismatch:
                    result[i] = False
                    continue

                # Lengths all match, check equality of values
                result[i] = np.array_equal(
                    self_values[self_start:self_stop],
                    other_values[other_start:other_stop]
                )
            return result
        else:
            raise ValueError("""
Cannot check equality of {typ} of length {a_len} with:
    {other}""".format(typ=type(self).__name__, a_len=len(self), other=repr(other)))

    @property
    def sindex(self):
        if self._sindex is None:
            self._sindex = HilbertRtree(self.bounds)
        return self._sindex

    @property
    def cx(self):
        """
        Geopandas-style spatial indexer to select a subset of the array by intersection
        with a bounding box

        Format of input should be ``.cx[xmin:xmax, ymin:ymax]``. Any of
        ``xmin``, ``xmax``, ``ymin``, and ``ymax`` can be provided, but input
        must include a comma separating x and y slices. That is, ``.cx[:, :]``
        will return the full series/frame, but ``.cx[:]`` is not implemented.
        """
        return _CoordinateIndexer(self)

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
            return self.copy() if copy else self

        if dtype == np.dtype('object'):
            return np.array(self, dtype='object')

        if isinstance(dtype, GeometryDtype):
            dtype = dtype.arrow_dtype.to_pandas_dtype()
        elif isinstance(dtype, pa.DataType):
            dtype = dtype.to_pandas_dtype()
        else:
            dtype = np.dtype(dtype)

        return self.__class__(np.asarray(self.data), dtype=dtype)

    astype.__doc__ = ExtensionArray.astype.__doc__

    def isna(self):
        return _extract_isnull_bytemap(self.data)

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
            if item.step is None or item.step == 1:
                # pyarrow only supports slice with step of 1
                return self.__class__(self.data[item], dtype=self.dtype)
            else:
                selected_indices = np.arange(len(self))[item]
                return self.take(selected_indices, allow_fill=False)
        elif isinstance(item, Iterable):
            item = np.asarray(item)
            if item.dtype == 'bool':
                # Convert to unsigned integer array of indices
                indices = np.nonzero(item)[0].astype(self.buffer_offsets[0].dtype)
                if len(indices):
                    return self.take(indices, allow_fill=False)
                else:
                    return self[:0]
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

        indices = np.asarray(indices)

        # Validate self non-empty (Pandas expects this error when array is empty)
        if len(self) == 0 and (not allow_fill or any(indices >= 0)):
            raise IndexError("cannot do a non-empty take on {typ}".format(
                typ=self.__class__.__name__,
            ))

        # Validate fill values
        if allow_fill and not (
                fill_value is None or
                np.isscalar(fill_value) and np.isnan(fill_value)):

            raise ValueError('non-None fill value not supported')

        # Validate indices
        invalid_mask = indices >= len(self)
        if not allow_fill:
            invalid_mask |= indices < -len(self)

        if any(invalid_mask):
            raise IndexError(
                "Index value out of bounds for {typ} of length {n}: "
                "{idx}".format(
                    typ=self.__class__.__name__,
                    n=len(self),
                    idx=indices[invalid_mask][0]
                )
            )

        if allow_fill:
            invalid_mask = indices < -1
            if any(invalid_mask):
                # ValueError expected by pandas ExtensionArray test suite
                raise ValueError(
                    "Invalid index value for {typ} with allow_fill=True: "
                    "{idx}".format(
                        typ=self.__class__.__name__,
                        n=len(self),
                        idx=indices[invalid_mask][0]
                    )
                )

            # Build pyarrow array of indices
            indices = pa.array(indices, mask=indices < 0)
        else:
            # Convert negative indices to positive
            negative_mask = indices < 0
            indices[negative_mask] = indices[negative_mask] + len(self)

            # Build pyarrow array of indices
            indices = pa.array(indices)

        return self.__class__(self.data.take(indices))

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

    def intersects_bounds(self, bounds, inds=None):
        """
        Test whether each element in the array intersects with the supplied bounds

        Args:
            bounds: Tuple of bounds coordinates of the form (x0, y0, x1, y1)
            inds: Optional array of indices into the array. If supplied, intersection
                calculations will be performed only on the elements selected by this
                array.  If not supplied, intersection calculations are performed
                on all elements.

        Returns:
            Array of boolean values indicating which elements of the array intersect
            with the supplied bounds
        """
        raise NotImplementedError()


class _BaseCoordinateIndexer(object):
    def __init__(self, sindex):
        self._sindex = sindex

    def _get_bounds(self, key):
        xs, ys = key
        # Handle xs and ys as scalar numeric values
        if type(xs) is not slice:
            xs = slice(xs, xs)
        if type(ys) is not slice:
            ys = slice(ys, ys)
        if xs.step is not None or ys.step is not None:
            raise ValueError(
                "Slice step not supported. The cx indexer uses slices to represent "
                "intervals in continuous coordinate space, and a slice step has no "
                "clear interpretation in this context."
            )
        xmin, ymin, xmax, ymax = self._sindex.total_bounds
        x0, y0, x1, y1 = (
            xs.start if xs.start is not None else xmin,
            ys.start if ys.start is not None else ymin,
            xs.stop if xs.stop is not None else xmax,
            ys.stop if ys.stop is not None else ymax,
        )
        # Handle inverted bounds
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return x0, x1, y0, y1

    def __getitem__(self, key):
        x0, x1, y0, y1 = self._get_bounds(key)
        covers_inds, overlaps_inds = self._sindex.covers_overlaps((x0, y0, x1, y1))
        return self._perform_get_item(covers_inds, overlaps_inds, x0, x1, y0, y1)

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        raise NotImplementedError()


class _CoordinateIndexer(_BaseCoordinateIndexer):
    def __init__(self, obj, parent=None):
        super(_CoordinateIndexer, self).__init__(obj.sindex)
        self._obj = obj
        self._parent = parent

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        overlaps_inds_mask = self._obj.intersects_bounds(
            (x0, y0, x1, y1), overlaps_inds
        )
        selected_inds = np.sort(
            np.concatenate([covers_inds, overlaps_inds[overlaps_inds_mask]])
        )
        if self._parent is not None:
            if len(self._parent) > 0:
                return self._parent.iloc[selected_inds]
            else:
                return self._parent
        else:
            return self._obj[selected_inds]


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
    if a1.dtype != np.object and a1.dtype != np.object:
        # a1 and a2 primitive
        return _lexographic_lt0(a1, a2)
    elif a1.dtype == np.object and a1.dtype == np.object:
        # a1 and a2 object, process recursively
        for e1, e2 in zip(a1, a2):
            if _lexographic_lt(e1, e2):
                return True
            elif _lexographic_lt(e2, e1):
                return False
        return len(a1) < len(a2)
    elif a1.dtype != np.object:
        # a2 is object array, a1 primitive
        return True
    else:
        # a1 is object array, a2 primitive
        return False


@ngjit
def _perform_extract_isnull_bytemap(bitmap, bitmap_length, bitmap_offset, dst_offset, dst):
    """
    Note: Copied from fletcher: See NOTICE for license info

    (internal) write the values of a valid bitmap as bytes to a pre-allocatored
    isnull bytemap.

    Parameters
    ----------
    bitmap: pyarrow.Buffer
        bitmap where a set bit indicates that a value is valid
    bitmap_length: int
        Number of bits to read from the bitmap
    bitmap_offset: int
        Number of bits to skip from the beginning of the bitmap.
    dst_offset: int
        Number of bytes to skip from the beginning of the output
    dst: numpy.array(dtype=bool)
        Pre-allocated numpy array where a byte is set when a value is null
    """
    for i in range(bitmap_length):
        idx = bitmap_offset + i
        byte_idx = idx // 8
        bit_mask = 1 << (idx % 8)
        dst[dst_offset + i] = (bitmap[byte_idx] & bit_mask) == 0


def _extract_isnull_bytemap(list_array):
    """
    Note: Copied from fletcher: See NOTICE for license info

    Extract the valid bitmaps of a chunked array into numpy isnull bytemaps.

    Parameters
    ----------
    chunked_array: pyarrow.ChunkedArray

    Returns
    -------
    valid_bytemap: numpy.array
    """
    result = np.zeros(len(list_array), dtype=bool)

    offset = 0
    chunk = list_array
    valid_bitmap = chunk.buffers()[0]
    if valid_bitmap:
        buf = memoryview(valid_bitmap)
        _perform_extract_isnull_bytemap(buf, len(chunk), chunk.offset, offset, result)
    else:
        return np.full(len(list_array), False)

    return result


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested1(
        fn, result, result_offset, values, value_offsets, missing
):
    assert len(value_offsets) == 1
    value_offsets0 = value_offsets[0]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            result[i + result_offset] = fn(values, value_offsets0[i:i + 2])


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested2(
        fn, result, result_offset, values, value_offsets, missing
):
    assert len(value_offsets) == 2
    value_offsets0 = value_offsets[0]
    value_offsets1 = value_offsets[1]
    n = len(value_offsets0) - 1
    for i in prange(n):
        if not missing[i]:
            start = value_offsets0[i]
            stop = value_offsets0[i + 1]
            result[i + result_offset] = fn(values, value_offsets1[start:stop + 1])


@jit(nogil=True, nopython=True, parallel=True)
def _geometry_map_nested3(
        fn, result, result_offset, values, value_offsets, missing
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
            result[i + result_offset] = fn(values, value_offsets2[start:stop + 1])


def is_geometry_array(data):
    """
    Check if the data is of geometry dtype.
    Does not include object array of Geometry/shapely scalars
    """
    if isinstance(getattr(data, "dtype", None), GeometryDtype):
        return True
    else:
        return False


def to_geometry_array(data, dtype=None):
    from . import (
        MultiPointArray,  LineArray, RingArray,
        MultiLineArray, PolygonArray, MultiPolygonArray
    )
    if sg is not None:
        shapely_to_spatialpandas = {
            sg.MultiPoint: MultiPointArray,
            sg.Point: MultiPointArray,
            sg.LineString: LineArray,
            sg.LinearRing: RingArray,
            sg.MultiLineString: MultiLineArray,
            sg.Polygon: PolygonArray,
            sg.MultiPolygon: MultiPolygonArray,
        }
    else:
        shapely_to_spatialpandas = {}

    err_msg = "Unable to convert data argument to a Geometry array"
    if is_geometry_array(data):
        # Keep data as is
        pass
    elif (is_array_like(data) or
            isinstance(data, (list, tuple))
            or gp and isinstance(data, (gp.GeoSeries, gp.array.GeometryArray))):

        if dtype is not None:
            data = dtype.construct_array_type()(data, dtype=dtype)
        elif len(data) == 0:
            raise ValueError(
                "Cannot infer spatialpandas geometry type from empty collection "
                "without dtype.\n"
            )
        else:
            # Check for list/array of geometry scalars.
            first_valid = None
            for val in data:
                if val is not None:
                    first_valid = val
                    break
            if isinstance(first_valid, Geometry):
                # Pass data to constructor of appropriate geometry array
                data = first_valid.construct_array_type()(data)
            elif type(first_valid) in shapely_to_spatialpandas:
                if isinstance(first_valid, sg.LineString):
                    # Handle mix of sg.LineString and sg.MultiLineString
                    for val in data:
                        if isinstance(val, sg.MultiLineString):
                            first_valid = val
                            break
                elif isinstance(first_valid, sg.Polygon):
                    # Handle mix of sg.Polygon and sg.MultiPolygon
                    for val in data:
                        if isinstance(val, sg.MultiPolygon):
                            first_valid = val
                            break

                array_type = shapely_to_spatialpandas[type(first_valid)]
                data = array_type(data)
            else:
                raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)
    return data
