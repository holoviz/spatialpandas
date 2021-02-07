import numpy as np
import pyarrow as pa
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry import Polygon
from ..geometry._algorithms.intersection import multipolygons_intersect_bounds
from ..geometry._algorithms.measures import compute_area, compute_line_length
from ..geometry._algorithms.orientation import orient_polygons
from ..geometry.base import GeometryDtype
from ..geometry.baselist import (
    GeometryList,
    GeometryListArray,
    _geometry_map_nested3,
)
from ..geometry.multiline import MultiLine, MultiLineArray


@register_extension_dtype
class MultiPolygonDtype(GeometryDtype):
    _geometry_name = 'multipolygon'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return MultiPolygonArray


class MultiPolygon(GeometryList):
    _nesting_levels = 2

    @classmethod
    def construct_array_type(cls):
        return MultiPolygonArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.MultiPolygon):
            multipolygon = []
            for polygon in shape:
                polygon_coords = Polygon._shapely_to_coordinates(polygon)
                multipolygon.append(polygon_coords)

            return multipolygon
        elif isinstance(shape, sg.Polygon):
            return [Polygon._shapely_to_coordinates(shape)]
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
        polygon_arrays = np.asarray(self.data.as_py())

        polygons = []
        for polygon_array in polygon_arrays:
            ring_arrays = [np.array(line_coords).reshape(len(line_coords) // 2, 2)
                           for line_coords in polygon_array]

            rings = [sg.LinearRing(ring_array) for ring_array in ring_arrays]
            polygons.append(sg.Polygon(shell=rings[0], holes=rings[1:]))

        return sg.MultiPolygon(polygons=polygons)

    @classmethod
    def from_shapely(cls, shape, orient=True):
        """
        Build a spatialpandas MultiPolygon object from a shapely shape

        Args:
            shape: A shapely Polygon or MultiPolygon shape
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    while there is a performance cost associated with this operation
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this convention is followed in the input data.
        Returns:
            spatialpandas MultiPolygon
        """
        import shapely.geometry as sg
        if orient:
            if isinstance(shape, sg.Polygon):
                shape = sg.polygon.orient(shape)
            elif isinstance(shape, sg.MultiPolygon):
                shape = sg.MultiPolygon([sg.polygon.orient(poly) for poly in shape])

        shape_parts = cls._shapely_to_coordinates(shape)
        return cls(shape_parts)

    @property
    def boundary(self):
        new_offsets = self.buffer_offsets[1]
        new_data = pa.ListArray.from_arrays(new_offsets, self.buffer_values)
        return MultiLine(new_data)

    @property
    def length(self):
        return compute_line_length(self.buffer_values, self.buffer_inner_offsets)

    @property
    def area(self):
        return compute_area(self.buffer_values, self.buffer_inner_offsets)

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        result = np.zeros(1, dtype=np.bool_)
        offsets1, offsets2 = self.buffer_offsets
        offsets0 = np.array([0, len(offsets1) - 1], dtype=np.uint32)
        multipolygons_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1), self.buffer_values,
            offsets0[:-1], offsets0[1:], offsets1, offsets2, result
        )
        return result[0]


class MultiPolygonArray(GeometryListArray):
    _element_type = MultiPolygon
    _nesting_levels = 3

    @property
    def _dtype_class(self):
        return MultiPolygonDtype

    @classmethod
    def from_geopandas(cls, ga, orient=True):
        """
        Build a spatialpandas MultiPolygonArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiPolygon or
                Polygon shapes.
            orient: If True (default), reorder polygon vertices so that outer shells
                    are stored in counter clockwise order and holes are stored in
                    clockwise order.  If False, accept vertices as given. Note that
                    while there is a performance cost associated with this operation
                    some algorithms will not behave properly if the above ordering
                    convention is not followed, so only set orient=False if it is
                    known that this convention is followed in the input data.

        Returns:
            MultiPolygonArray
        """
        mpa = super().from_geopandas(ga)
        if orient:
            return mpa.oriented()
        else:
            return mpa

    def oriented(self):
        missing = np.concatenate([self.isna(), [False]])
        buffer_values = self.buffer_values.copy()
        multipoly_offsets, poly_offsets, ring_offsets = self.buffer_offsets

        orient_polygons(buffer_values, poly_offsets, ring_offsets)

        pa_rings = pa.ListArray.from_arrays(
            pa.array(ring_offsets), pa.array(buffer_values)
        )
        pa_polys = pa.ListArray.from_arrays(
            pa.array(poly_offsets), pa_rings,
        )
        pa_multipolys = pa.ListArray.from_arrays(
            pa.array(multipoly_offsets, mask=missing), pa_polys
        )
        return self.__class__(pa_multipolys)

    @property
    def boundary(self):
        offsets = self.buffer_offsets
        inner_data = pa.ListArray.from_arrays(offsets[2], self.buffer_values)
        new_data = pa.ListArray.from_arrays(offsets[1][offsets[0]], inner_data)
        return MultiLineArray(new_data)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        _geometry_map_nested3(
            compute_line_length,
            result,
            self.buffer_values,
            self.buffer_offsets,
            self.isna(),
        )
        return result

    @property
    def area(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        _geometry_map_nested3(
            compute_area,
            result,
            self.buffer_values,
            self.buffer_offsets,
            self.isna(),
        )
        return result

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        offsets0, offsets1, offsets2 = self.buffer_offsets
        start_offsets0 = offsets0[:-1]
        stop_offsets0 = offsets0[1:]
        if inds is not None:
            start_offsets0 = start_offsets0[inds]
            stop_offsets0 = stop_offsets0[inds]

        result = np.zeros(len(start_offsets0), dtype=np.bool_)
        multipolygons_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1), self.buffer_values,
            start_offsets0, stop_offsets0, offsets1, offsets2, result
        )
        return result


def _multi_polygon_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return MultiPolygonArray([
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
    make_array_nonempty.register(MultiPolygonDtype)(_multi_polygon_array_non_empty)
