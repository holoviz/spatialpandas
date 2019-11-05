from __future__ import absolute_import
from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry import Polygon2d
from spatialpandas.geometry.base import (
    GeometryArray, GeometryDtype, Geometry2
)
from spatialpandas.geometry.multiline2d import MultiLine2dArray, MultiLine2d
import numpy as np
from spatialpandas.geometry._algorithms import (
    compute_line_length, compute_area, geometry_map_nested3
)
from dask.dataframe.extensions import make_array_nonempty
import pyarrow as pa


@register_extension_dtype
class MultiPolygon2dDtype(GeometryDtype):
    _geometry_name = 'multipolygon2d'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiPolygon2dArray


class MultiPolygon2d(Geometry2):

    @classmethod
    def _shapely_to_coordinates(cls, shape, orient=True):
        import shapely.geometry as sg
        if isinstance(shape, sg.MultiPolygon):
            multipolygon = []
            for polygon in shape:
                polygon_coords = Polygon2d._shapely_to_coordinates(polygon, orient)
                multipolygon.append(polygon_coords)

            return multipolygon
        elif isinstance(shape, sg.Polygon):
            return [Polygon2d._shapely_to_coordinates(shape, orient)]
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Polygon or MultiPolygon
""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely MultiPolygon shape
        """
        import shapely.geometry as sg
        polygon_arrays = np.asarray(self.data)

        polygons = []
        for polygon_array in polygon_arrays:
            ring_arrays = [line_coords.reshape(len(line_coords) // 2, 2)
                           for line_coords in polygon_array]

            rings = [sg.LinearRing(ring_array) for ring_array in ring_arrays]
            polygons.append(sg.Polygon(shell=rings[0], holes=rings[1:]))

        return sg.MultiPolygon(polygons=polygons)

    @classmethod
    def from_shapely(cls, shape, orient=True):
        """
        Build a spatialpandas MultiPolygon2d object from a shapely shape

        Args:
            shape: A shapely Polygon or MultiPolygon shape
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this conventions is followed in the input data.
        Returns:
            spatialpandas MultiPolygon2d
        """
        shape_parts = cls._shapely_to_coordinates(shape, orient)
        return cls(shape_parts)


    @property
    def boundary(self):
        new_offsets = self.buffer_offsets[1]
        new_data = pa.ListArray.from_arrays(new_offsets, self.buffer_values)
        return MultiLine2d(new_data)

    @property
    def length(self):
        return compute_line_length(self._values, self._value_offsets)

    @property
    def area(self):
        return compute_area(self._values, self._value_offsets)


class MultiPolygon2dArray(GeometryArray):
    _element_type = MultiPolygon2d
    _nesting_levels = 3

    @property
    def _dtype_class(self):
        return MultiPolygon2dDtype

    @classmethod
    def from_geopandas(cls, ga, orient=True):
        """
        Build a spatialpandas MultiPolygon2dArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiPolygon or
                Polygon shapes.
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this conventions is followed in the input data.

        Returns:
            MultiPolygon2dArray
        """
        return cls([MultiPolygon2d._shapely_to_coordinates(shape, orient) for shape in ga])

    @property
    def boundary(self):
        offsets = self.buffer_offsets
        inner_data = pa.ListArray.from_arrays(offsets[2], self.buffer_values)
        new_data = pa.ListArray.from_arrays(offsets[1][offsets[0]], inner_data)
        return MultiLine2dArray(new_data)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        for c, result_offset in enumerate(self.offsets):
            geometry_map_nested3(
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
            geometry_map_nested3(
                compute_area,
                result,
                result_offset,
                self.buffer_values,
                self.buffer_offsets,
                self.isna(),
            )
        return result


def multi_polygon_array_non_empty(dtype):
    return MultiPolygon2dArray([
        [
            [[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0],
             [1.1, 1.1, 1.5, 1.9, 1.9, 1.1, 1.1, 1.1]]
        ],
        [
            [[0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.5, 3.0, -1.0, 1.0, 0.0, 0.0],
             [0.2, 0.2, 0.5, 1.0, 0.8, 0.2, 0.2, 0.2],
             [0.5, 1.25, 0.3, 2.0, 0.8, 2.0, 0.5, 1.25]]
        ]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(MultiPolygon2dDtype)(multi_polygon_array_non_empty)
