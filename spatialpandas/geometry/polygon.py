from __future__ import absolute_import
from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry1
)
from spatialpandas.geometry.multiline import MultiLineArray, MultiLine
import numpy as np
from spatialpandas.geometry._algorithms import (
    compute_line_length, compute_area, geometry_map_nested2
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class PolygonDtype(GeometryDtype):
    _geometry_name = 'polygon'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return PolygonArray


class Polygon(Geometry1):
    @classmethod
    def _shapely_to_coordinates(cls, shape, orient=True):
        import shapely.geometry as sg
        if isinstance(shape, sg.Polygon):
            if orient:
                shape = sg.polygon.orient(shape)
            exterior = np.asarray(shape.exterior.ctypes)
            polygon_coords = [exterior]
            for ring in shape.interiors:
                interior = np.asarray(ring.ctypes)
                polygon_coords.append(interior)

            return polygon_coords
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Polygon
""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely Polygon shape
        """
        import shapely.geometry as sg
        ring_arrays = [line_coords.reshape(len(line_coords) // 2, 2)
                       for line_coords in np.asarray(self.data)]
        rings = [sg.LinearRing(ring_array) for ring_array in ring_arrays]
        return sg.Polygon(shell=rings[0], holes=rings[1:])

    @classmethod
    def from_shapely(cls, shape, orient=True):
        """
        Build a spatialpandas Polygon object from a shapely shape

        Args:
            shape: A shapely Polygon shape
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    while there is a performance cost associated with this operation
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this convention is followed in the input data.
        Returns:
            spatialpandas Polygon
        """
        shape_parts = cls._shapely_to_coordinates(shape, orient)
        return cls(shape_parts)

    @property
    def boundary(self):
        # The representation of PolygonArray and MultiLineArray is identical
        return MultiLine(self.data)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return compute_area(self._values, self._value_offsets)


class PolygonArray(GeometryArray):
    _element_type = Polygon
    _nesting_levels = 2

    @property
    def _dtype_class(self):
        return PolygonDtype

    @classmethod
    def from_geopandas(cls, ga, orient=True):
        """
        Build a spatialpandas PolygonArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of Polygon shapes.
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    while there is a performance cost associated with this operation
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this convention is followed in the input data.
        Returns:
            PolygonArray
        """
        return cls([Polygon._shapely_to_coordinates(shape, orient) for shape in ga])

    @property
    def boundary(self):
        # The representation of PolygonArray and MultiLineArray is identical
        return MultiLineArray(self.data)

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
        result = np.full(len(self), np.nan, dtype=np.float64)
        for c, result_offset in enumerate(self.offsets):
            geometry_map_nested2(
                compute_area,
                result,
                result_offset,
                self.buffer_values,
                self.buffer_offsets,
                self.isna(),
            )
        return result


def _polygon_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return PolygonArray(
        [
            [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0],
             [1.1, 1.1, 1.5, 1.9, 1.9, 1.1, 1.1, 1.1]],
            [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0]]
        ], dtype=dtype
    )


if make_array_nonempty:
    make_array_nonempty.register(PolygonDtype)(_polygon_array_non_empty)
