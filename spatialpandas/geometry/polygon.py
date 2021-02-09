import numpy as np
import pyarrow as pa
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry._algorithms.intersection import polygons_intersect_bounds
from ..geometry._algorithms.measures import compute_area, compute_line_length
from ..geometry._algorithms.orientation import orient_polygons
from ..geometry.base import GeometryDtype
from ..geometry.baselist import (
    GeometryList,
    GeometryListArray,
    _geometry_map_nested2,
)
from ..geometry.multiline import MultiLine, MultiLineArray


@register_extension_dtype
class PolygonDtype(GeometryDtype):
    _geometry_name = 'polygon'

    @classmethod
    def construct_array_type(cls, *args):
        if len(args) > 0:
            raise NotImplementedError("construct_array_type does not support arguments")
        return PolygonArray


class Polygon(GeometryList):
    _nesting_levels = 1

    @classmethod
    def construct_array_type(cls):
        return PolygonArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.Polygon):
            if shape.exterior is not None:
                exterior = np.asarray(shape.exterior.ctypes)
                polygon_coords = [exterior]
            else:
                polygon_coords = [np.array([])]
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
        ring_arrays = [np.asarray(line_coords).reshape(len(line_coords) // 2, 2)
                       for line_coords in np.asarray(self.data.as_py())]
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
        import shapely.geometry as sg
        if orient:
            shape = sg.polygon.orient(shape)

        shape_parts = cls._shapely_to_coordinates(shape)
        return cls(shape_parts)

    @property
    def boundary(self):
        # The representation of PolygonArray and MultiLineArray is identical
        return MultiLine(self.data)

    @property
    def length(self):
        return compute_line_length(self.buffer_values, self.buffer_inner_offsets)

    @property
    def area(self):
        return compute_area(self.buffer_values, self.buffer_inner_offsets)

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        result = np.zeros(1, dtype=np.bool_)
        offsets1 = self.buffer_inner_offsets
        start_offsets0 = np.array([0], dtype=np.uint32)
        stop_offsets0 = np.array([len(offsets1) - 1], dtype=np.uint32)
        polygons_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets0, stop_offsets0, offsets1, result
        )
        return result[0]


class PolygonArray(GeometryListArray):
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
        polygons = super().from_geopandas(ga)
        if orient:
            return polygons.oriented()
        else:
            return polygons

    def oriented(self):
        missing = np.concatenate([self.isna(), [False]])
        buffer_values = self.buffer_values.copy()
        poly_offsets, ring_offsets = self.buffer_offsets

        orient_polygons(buffer_values, poly_offsets, ring_offsets)

        pa_rings = pa.ListArray.from_arrays(
            pa.array(ring_offsets), pa.array(buffer_values)
        )
        pa_polys = pa.ListArray.from_arrays(
            pa.array(poly_offsets, mask=missing), pa_rings,
        )

        return self.__class__(pa_polys)

    @property
    def boundary(self):
        # The representation of PolygonArray and MultiLineArray is identical
        return MultiLineArray(self.data)

    @property
    def length(self):
        result = np.full(len(self), np.nan, dtype=np.float64)
        _geometry_map_nested2(
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
        _geometry_map_nested2(
            compute_area,
            result,
            self.buffer_values,
            self.buffer_offsets,
            self.isna(),
        )
        return result

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        offsets0, offsets1 = self.buffer_offsets
        start_offsets0 = offsets0[:-1]
        stop_offsets0 = offsets0[1:]
        if inds is not None:
            start_offsets0 = start_offsets0[inds]
            stop_offsets0 = stop_offsets0[inds]

        result = np.zeros(len(start_offsets0), dtype=np.bool_)
        polygons_intersect_bounds(
            float(x0), float(y0), float(x1), float(y1),
            self.buffer_values, start_offsets0, stop_offsets0, offsets1, result
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
