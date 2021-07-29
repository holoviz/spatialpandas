import re
from collections.abc import Container, Iterable
from numbers import Integral

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.types import is_array_like

from .._optional_imports import gp, sg
from ..spatialindex import HilbertRtree
from ..spatialindex.rtree import _distances_from_bounds
from ..utils import ngjit


def _unwrap_geometry(a, element_dtype):
    try:
        if np.isscalar(a) and np.isnan(a):
            # replace top-level nana with None
            return None
    except (TypeError, ValueError):
        # Not nan, continue
        pass
    if isinstance(a, Geometry):
        return a.data.as_py()
    elif sg and isinstance(a, sg.base.BaseGeometry):
        return element_dtype._shapely_to_coordinates(a)
    else:
        return a


class GeometryDtype(ExtensionDtype):
    _geometry_name = 'geometry'
    base = np.dtype('O')
    _metadata = ('subtype',)
    na_value = np.nan

    def __from_arrow__(self, data):
        return self.construct_array_type()(data, dtype=self)

    @classmethod
    def _arrow_element_type_from_numpy_subtype(cls, subtype):
        raise NotImplementedError

    @classmethod
    def construct_array_type(cls, *args):
        return GeometryArray

    @classmethod
    def _parse_subtype(cls, dtype_string):
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
    def construct_from_string(cls, string):
        # lowercase string
        try:
            string = string.lower()
            if not isinstance(string, str):
                raise AttributeError
        except AttributeError:
            raise TypeError(
                "'construct_from_string' expects a string, got {typ}".format(
                    typ=type(string)))

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

    def __init__(self, subtype):
        if isinstance(subtype, GeometryDtype):
            self.subtype = subtype.subtype
        else:
            self.subtype = np.dtype(subtype)

        # Validate the subtype is numeric
        if self.subtype.kind not in ('i', 'u', 'f'):
            raise ValueError("Received non-numeric type of kind '{}'".format(self.kind))

        array_type = self.construct_array_type()
        self.arrow_dtype = array_type._arrow_type_from_numpy_element_dtype(subtype)

    def __hash__(self):
        return hash((self.__class__, self.arrow_dtype))

    def __str__(self):
        return "{}[{}]".format(self._geometry_name, str(self.subtype.name))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,  str(self.subtype.name))

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


class Geometry:
    def __init__(self, data, dtype=None):
        if isinstance(data, pa.Scalar):
            # Use arrow Scalar as is
            self.data = data
        else:
            # Convert to arrow Scalar
            self.data = pa.array([data])[0]

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.data.as_py())

    def __hash__(self):
        return hash((self.__class__, np.array(self.data.as_py()).tobytes()))

    def __eq__(self, other):
        if isinstance(other, Container):
            return other == self
        if type(other) is not type(self):
            return False
        return self.data == other.data

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

    def intersects_bounds(self, bounds):
        raise NotImplementedError()

    def intersects(self, shape):
        raise NotImplementedError(
            "intersects not yet implemented for %s objects" % type(self).__name__
        )


class GeometryArray(ExtensionArray):
    _element_type = Geometry
    _can_hold_na = True

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
        return cls([
            cls._element_type._shapely_to_coordinates(shape)
            if shape is not None else None
            for shape in ga
        ])

    def to_geopandas(self):
        """
        Convert a spatialpandas geometry array into a geopandas GeometryArray

        Returns:
            geopandas GeometryArray
        """
        from geopandas.array import from_shapely
        return from_shapely([el.to_shapely() for el in self])

    # Constructor
    def __init__(self, array, dtype=None, copy=None):
        # Choose default dtype for empty arrays
        try:
            if len(array) == 0 and dtype is None:
                dtype = 'float64'
        except:
            # len failed
            pass

        # See if we can determine arrow array type
        if isinstance(dtype, GeometryDtype):
            # Use arrow type as-is
            arrow_dtype = dtype.arrow_dtype
        elif isinstance(dtype, pa.DataType):
            arrow_dtype = dtype
        elif dtype is not None and dtype != np.dtype('object'):
            # Scalar element dtype
            arrow_dtype = self._arrow_type_from_numpy_element_dtype(dtype)
        else:
            # Let arrow infer type
            arrow_dtype = None

        # Unwrap GeometryList elements to numpy arrays
        if is_array_like(array) or isinstance(array, list):
            array = [_unwrap_geometry(el, self._element_type) for el in array]
            array = pa.array(array, type=arrow_dtype)
        elif isinstance(array, pa.Array):
            # Nothing to do
            pass
        elif isinstance(array, pa.ChunkedArray):
            array = pa.concat_arrays(array.chunks)
        else:
            raise ValueError(
                "Unsupported type passed for {}: {}".format(
                    self.__class__.__name__, type(array)
                )
            )

        # Save off pyarrow array
        self.data = array

        # Compute types
        np_type = self._numpy_element_dtype_from_arrow_type(self.data.type)
        self._numpy_element_type = np.dtype(np_type)
        self._dtype = self._dtype_class(np_type)

        # Initialize backing property for spatial index
        self._sindex = None

    @classmethod
    def _arrow_type_from_numpy_element_dtype(cls, dtype):
        raise NotImplementedError

    def _numpy_element_dtype_from_arrow_type(self, pyarrow_type):
        raise NotImplementedError

    @property
    def _dtype_class(self):
        return GeometryDtype

    @property
    def numpy_dtype(self):
        return self._numpy_element_type

    # Arrow conversion
    def __arrow_array__(self, type=None):
        return self.data

    # ExtensionArray methods
    @property
    def dtype(self):
        return self._dtype

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

    @property
    def nbytes(self):
        size = 0
        for buf in self.data.buffers():
            if buf is not None:
                size += buf.size
        return size

    def isna(self):
        return _extract_isnull_bytemap(self.data)

    isna.__doc__ = ExtensionArray.isna.__doc__

    def copy(self):
        return type(self)(self.data, self.dtype)

    copy.__doc__ = ExtensionArray.copy.__doc__

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
            for i in range(len(self)):
                result[i] = self[i] == other[i]
            return result
        if isinstance(other, (self.dtype.type, type(None))):
            result = np.zeros(len(self), dtype=np.bool_)
            for i in range(len(self)):
                result[i] = self[i] == other
            return result
        raise ValueError("""
Cannot check equality of {typ} of length {a_len} with:
    {other}""".format(typ=type(self).__name__, a_len=len(self), other=repr(other)))

    def __contains__(self, item) -> bool:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        err_msg = ("Only integers, slices and integer or boolean"
                   "arrays are valid indices.")
        if isinstance(item, tuple) and len(item) == 2:
            if item[0] is Ellipsis:
                item = item[1]
            elif item[1] is Ellipsis:
                item = item[0]

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
                    return self._element_type(value, self.numpy_dtype)
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
            if isinstance(item, (np.ndarray, ExtensionArray)):
                # Leave numpy and pandas arrays alone
                kind = item.dtype.kind
            else:
                item = pd.array(item)
                kind = item.dtype.kind

            if len(item) == 0:
                return self.take([], allow_fill=False)
            elif kind == 'b':
                # Check mask length is compatible
                if len(item) != len(self):
                    raise IndexError(
                        "Boolean index has wrong length: {} instead of {}"
                        .format(len(item), len(self))
                    )

                # check for NA values
                if any(pd.isna(item)):
                    raise ValueError(
                        "Cannot mask with a boolean indexer containing NA values"
                    )

                # Convert to unsigned integer array of indices
                indices = np.nonzero(item)[0].astype(np.uint32)
                if len(indices):
                    return self.take(indices, allow_fill=False)
                else:
                    return self[:0]
            elif kind in ('i', 'u'):
                if any(pd.isna(item)):
                    raise ValueError(
                        "Cannot index with an integer indexer containing NA values"
                    )
                return self.take(item, allow_fill=False)
            else:
                raise IndexError(err_msg)
        else:
            raise IndexError(err_msg)

    def take(self, indices, allow_fill=False, fill_value=None):
        indices = np.asarray(indices)

        # Validate self non-empty (Pandas expects this error when array is empty)
        if (len(self) == 0 and len(indices) > 0 and
                (not allow_fill or any(indices >= 0))):
            raise IndexError("cannot do a non-empty take from an empty axes|out of bounds on {typ}".format(
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
                        idx=indices[invalid_mask][0]
                    )
                )

            # Build pyarrow array of indices
            indices = pa.array(indices.astype('int'), mask=indices < 0)
        else:
            # Convert negative indices to positive
            negative_mask = indices < 0
            indices[negative_mask] = indices[negative_mask] + len(self)

            # Build pyarrow array of indices
            indices = pa.array(indices.astype('int'))

        return self.__class__(self.data.take(indices), dtype=self.dtype)

    take.__doc__ = ExtensionArray.take.__doc__

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=None):
        if isinstance(scalars, cls):
            return scalars
        elif isinstance(scalars, Geometry):
            scalars = [scalars]

        return cls([
            None if np.isscalar(v) and np.isnan(v) else v for v in scalars
        ], dtype=dtype)

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
        from pandas.core.missing import get_fill_func
        from pandas.util._validators import validate_fillna_kwargs

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
                func = get_fill_func(method)
                new_values = func(self.astype(object), limit=limit, mask=mask)
                # From pandas 1.3, get_fill_func also return mask
                new_values = new_values[0] if isinstance(new_values, tuple) else new_values
                new_values = self._from_sequence(new_values, self._dtype)
            else:
                # fill with value
                new_values = np.asarray(self)
                if isinstance(value, Geometry):
                    value = [value]
                new_values[mask] = value
                new_values = self.__class__(new_values, dtype=self.dtype)
        else:
            new_values = self.copy()
        return new_values

    fillna.__doc__ = ExtensionArray.fillna.__doc__

    # Geometry properties/methods
    @property
    def sindex(self):
        if self._sindex is None:
            self.build_sindex()
        return self._sindex

    def build_sindex(self, **kwargs):
        if self._sindex is None:
            self._sindex = HilbertRtree(self.bounds, **kwargs)
        return self

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
    def total_bounds(self):
        raise NotImplementedError

    @property
    def total_bounds_x(self):
        raise NotImplementedError

    @property
    def total_bounds_y(self):
        raise NotImplementedError

    @property
    def bounds(self):
        raise NotImplementedError

    def hilbert_distance(self, total_bounds=None, p=10):
        # Handle default total_bounds
        if total_bounds is None:
            total_bounds = list(self.total_bounds)

        # Expand zero width bounds
        if total_bounds[0] == total_bounds[2]:
            total_bounds[2] += 1.0
        if total_bounds[1] == total_bounds[3]:
            total_bounds[3] += 1.0
        total_bounds = tuple(total_bounds)

        return _distances_from_bounds(self.bounds, total_bounds, p)

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

    def intersects(self, shape, inds=None):
        """
        Test whether each element in the array intersects with the supplied shape

        Args:
            shape: The spatialpandas shape to compute intersections with
            inds: Optional array of indices into the array. If supplied, intersection
                calculations will be performed only on the elements selected by this
                array.  If not supplied, intersection calculations are performed
                on all elements.

        Returns:
            Array of boolean values indicating which elements of the array intersect
            with the supplied shape
        """
        raise NotImplementedError(
            "intersects not yet implemented for %s objects" % type(self).__name__
        )


class _BaseCoordinateIndexer:
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
        if self._sindex:
            xmin, ymin, xmax, ymax = self._sindex.total_bounds
        else:
            xmin, ymin, xmax, ymax = self._obj.total_bounds
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
        if self._sindex:
            covers_inds, overlaps_inds = self._sindex.covers_overlaps((x0, y0, x1, y1))
        else:
            covers_inds, overlaps_inds = None, None
        return self._perform_get_item(covers_inds, overlaps_inds, x0, x1, y0, y1)

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        raise NotImplementedError()


class _CoordinateIndexer(_BaseCoordinateIndexer):
    def __init__(self, obj, parent=None):
        super().__init__(obj._sindex)
        self._obj = obj
        self._parent = parent

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        overlaps_inds_mask = self._obj.intersects_bounds(
            (x0, y0, x1, y1), overlaps_inds
        )
        if covers_inds is not None:
            selected_inds = np.sort(
                np.concatenate([covers_inds, overlaps_inds[overlaps_inds_mask]])
            )
            if self._parent is not None:
                if len(self._parent) > 0:
                    return self._parent.iloc[selected_inds]
                else:
                    return self._parent
            return self._obj[selected_inds]
        else:
            if self._parent is not None:
                if len(self._parent) > 0:
                    return self._parent[overlaps_inds_mask]
                else:
                    return self._parent
            return self._obj[overlaps_inds_mask]


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


def is_geometry_array(data):
    """
    Check if the data is of geometry dtype.
    Does not include object array of GeometryList/shapely scalars
    """
    if isinstance(getattr(data, "dtype", None), GeometryDtype):
        return True
    else:
        return False


def to_geometry_array(data, dtype=None):
    from . import (LineArray, MultiLineArray, MultiPointArray,
                   MultiPolygonArray, PointArray, PolygonArray, RingArray)
    if sg is not None:
        shapely_to_spatialpandas = {
            sg.Point: PointArray,
            sg.MultiPoint: MultiPointArray,
            sg.LineString: LineArray,
            sg.LinearRing: RingArray,
            sg.MultiLineString: MultiLineArray,
            sg.Polygon: PolygonArray,
            sg.MultiPolygon: MultiPolygonArray,
        }
    else:
        shapely_to_spatialpandas = {}

    # Normalize dtype from string
    if dtype is not None:
        dtype = pd.array([], dtype=dtype).dtype

    err_msg = "Unable to convert data argument to a GeometryList array"
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
                data = array_type.from_geopandas(data)
            else:
                raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)
    return data
