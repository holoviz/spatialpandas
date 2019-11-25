from functools import total_ordering

import pyarrow as pa
import numpy as np
from spatialpandas.geometry import Geometry, GeometryArray, GeometryDtype


@total_ordering
class GeometryStruct(Geometry):
    def __repr__(self):
        flat_data = self._flat_data()
        return "{}({})".format(self.__class__.__name__, flat_data)

    def _flat_data(self):
        pydata = self.data.as_py()
        n = len(pydata) // 2
        view_data = [pydata[f + str(i)] for i in range(n) for f in ['x', 'y']]
        return view_data

    def __hash__(self):
        return hash((self.__class__, tuple(self._flat_data())))

    def __lt__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return tuple(self._flat_data()) < tuple(other._flat_data())


class GeometryStructArray(GeometryArray):
    _num_coords = 1
    _element_type = GeometryStruct

    @classmethod
    def _arrow_type_from_numpy_element_dtype(cls, dtype):
        # Scalar element dtype
        arrow_dtype = pa.from_numpy_dtype(dtype)
        return pa.struct([(field, arrow_dtype) for field in cls.fields()])

    @classmethod
    def _numpy_element_dtype_from_arrow_type(cls, pyarrow_type):
        return pyarrow_type[0].type.to_pandas_dtype()

    @classmethod
    def fields(cls):
        return [
            d + str(i) for i in range(cls._num_coords) for d in ['x', 'y']
        ]

    def __init__(self, array, dtype=None):
        def invalid_array():
            err_msg = (
                "Invalid array with type {typ}\n"
                "A {cls} may constructed from:\n"
                "    - A 1-d array with length divisible by {n} of interleaved\n"
                "      x y coordinates\n"
                "    - A tuple of {n} 1-d arrays\n"
                "    - A list of dictionaries where each has the keys: {fields}.\n"
                "      In this case the dtype argument must also be provided.\n"
            ).format(
                typ=type(array), cls=self.__class__.__name__,
                n=2 * self._num_coords, fields=self.fields()
            )
            raise ValueError(err_msg)

        if isinstance(dtype, GeometryDtype):
            dtype = dtype.subtype

        if isinstance(array, pa.Array):
            if not isinstance(array, pa.StructArray):
                invalid_array()
        elif isinstance(array, tuple):
            if len(array) == 2 * self._num_coords:
                if dtype:
                    [array[i].astype(dtype) for i in range(len(array))]
                array = pa.StructArray.from_arrays(array, names=self.fields())
            else:
                invalid_array()
        else:
            array = np.asarray(array).ravel()
            if array.dtype.kind == 'O':
                fields = self.fields()
                for i in range(len(array)):
                    el = array[i]
                    if isinstance(el, (list, np.ndarray)) and len(el) == len(fields):
                        array[i] = {f: el[i] for i, f in enumerate(fields)}
                if dtype is not None:
                    patype = pa.from_numpy_dtype(dtype)
                    dtype = pa.struct([(f, patype) for f in self.fields()])
            else:
                if dtype:
                    array = array.astype(dtype)
                if len(array) % (2 * self._num_coords) == 0:
                    # interleaved
                    array = array.reshape(
                        (len(array) // (2 * self._num_coords),
                         (2 * self._num_coords))
                    )
                    arrays = [array[:, i] for i in range(2 * self._num_coords)]
                    array = pa.StructArray.from_arrays(arrays, names=self.fields())
                else:
                    invalid_array()

        super(GeometryStructArray, self).__init__(array, dtype)

        # Validate arrow type
        arrow_type = self.data.type

        def invalid_type():
            err_msg = (
                "Invalid element type {}\n"
                "Expected StructType with fields: {}"
            ).format(arrow_type, self.fields())

            raise ValueError(err_msg)

        if not isinstance(arrow_type, pa.StructType):
            invalid_type()
        elif arrow_type.num_children != self._num_coords * 2:
            invalid_type()
        for i in range(self._num_coords):
            if arrow_type[2 * i].name != 'x' + str(i):
                invalid_type()
            if arrow_type[2 * i + 1].name != 'y' + str(i):
                invalid_type()
