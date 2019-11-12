from __future__ import absolute_import
from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry1
)
from spatialpandas.geometry.multiline2d import MultiLine2dArray, MultiLine2d
import numpy as np
from spatialpandas.geometry._algorithms import (
    compute_line_length, compute_area, geometry_map_nested2
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class Polygon2dDtype(GeometryDtype):
    _geometry_name = 'polygon2d'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return Polygon2dArray


class Polygon2d(Geometry1):
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
        Build a spatialpandas Polygon2d object from a shapely shape

        Args:
            shape: A shapely Polygon shape
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this conventions is followed in the input data.
        Returns:
            spatialpandas Polygon2d
        """
        shape_parts = cls._shapely_to_coordinates(shape, orient)
        return cls(shape_parts)

    @property
    def boundary(self):
        # The representation of Polygon2dArray and MultiLine2dArray is identical
        return MultiLine2d(self.data)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return compute_area(self._values, self._value_offsets)


class Polygon2dArray(GeometryArray):
    _element_type = Polygon2d
    _nesting_levels = 2

    @property
    def _dtype_class(self):
        return Polygon2dDtype

    @classmethod
    def from_geopandas(cls, ga, orient=True):
        """
        Build a spatialpandas Polygon2dArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of Polygon shapes.
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this convention is followed in the input data.
        Returns:
            Polygon2dArray
        """
        return cls([Polygon2d._shapely_to_coordinates(shape, orient) for shape in ga])

    @property
    def boundary(self):
        # The representation of Polygon2dArray and MultiLine2dArray is identical
        return MultiLine2dArray(self.data)

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


def polygon_array_non_empty(dtype):
    return Polygon2dArray(
        [
            [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0],
             [1.1, 1.1, 1.5, 1.9, 1.9, 1.1, 1.1, 1.1]],
            [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0]]
        ], dtype=dtype
    )


if make_array_nonempty:
    make_array_nonempty.register(Polygon2dDtype)(polygon_array_non_empty)
