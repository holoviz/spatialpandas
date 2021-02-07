import numpy as np
from dask.dataframe.extensions import make_array_nonempty
from pandas.core.dtypes.dtypes import register_extension_dtype

from ..geometry._algorithms.intersection import (point_intersects_polygon,
                                                 segment_intersects_point)
from ..geometry.base import GeometryDtype
from ..geometry.basefixed import GeometryFixed, GeometryFixedArray
from ..utils import ngpjit


@register_extension_dtype
class PointDtype(GeometryDtype):
    _geometry_name = 'point'
    @classmethod
    def construct_array_type(cls, *args):
        return PointArray


class Point(GeometryFixed):

    @classmethod
    def construct_array_type(cls):
        return PointArray

    @classmethod
    def _shapely_to_coordinates(cls, shape):
        import shapely.geometry as sg
        if isinstance(shape, sg.Point):
            # Single point
            return np.asarray(shape.ctypes)
        else:
            raise ValueError("""
Received invalid value of type {typ}. Must be an instance of Point,
or MultiPoint""".format(typ=type(shape).__name__))

    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely MultiPoint shape
        """
        import shapely.geometry as sg
        return sg.Point(self.flat_values)

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas Point object from a shapely shape

        Args:
            shape: A shapely MultiPoint or Point shape

        Returns:
            spatialpandas MultiPoint
        """
        return super().from_shapely(shape)

    @property
    def x(self):
        return self.flat_values[0]

    @property
    def y(self):
        return self.flat_values[1]

    @property
    def length(self):
        return 0.0

    @property
    def area(self):
        return 0.0

    def intersects_bounds(self, bounds):
        x0, y0, x1, y1 = bounds
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        outside = (np.isnan(self.x) or
                   self.x < x0 or self.x > x1 or
                   self.y < y0 or self.y > y1)

        return not outside

    def _intersects_point(self, point):
        return self.x == point.x and self.y == point.y

    def _intersects_multipoint(self, multipoint):
        flat = multipoint.flat_values
        return np.any((self.x == flat[0::2]) & (self.y == flat[1::2]))

    def _intersects_line(self, line):
        buffer_values = line.buffer_values
        offsets = line.buffer_inner_offsets

        for i in range(len(offsets) - 1):
            start = offsets[i]
            stop = offsets[i + 1]
            flat = buffer_values[start:stop]
            xs = flat[0::2]
            ys = flat[1::2]
            bounds = (min(xs), min(ys), max(xs), max(ys))

            # Check bounding box
            intersects_bounds = self.intersects_bounds(bounds)
            if not intersects_bounds:
                continue

            # Check if point exactly intersects vertex of line
            intersects_vert = np.any((self.x == flat[0::2]) & (self.y == flat[1::2]))
            if intersects_vert:
                return True

            # Check if point is exactly on one of the segments of the line
            bx = self.x
            by = self.y
            for j in range(len(xs) - 1):
                ax0 = xs[j]
                ay0 = ys[j]
                ax1 = xs[j + 1]
                ay1 = ys[j + 1]
                if segment_intersects_point(ax0, ay0, ax1, ay1, bx, by):
                    return True

        return False

    def _intersects_polygon(self, polygon):
        return point_intersects_polygon(
            self.x, self.y, polygon.buffer_values, polygon.buffer_inner_offsets
        )

    def intersects(self, shape):
        from . import Line, MultiLine, MultiPoint, MultiPolygon, Polygon
        if isinstance(shape, Point):
            return self._intersects_point(shape)
        elif isinstance(shape, MultiPoint):
            return self._intersects_multipoint(shape)
        elif isinstance(shape, Line):
            return self._intersects_line(shape)
        elif isinstance(shape, MultiLine):
            return self._intersects_line(shape)
        elif isinstance(shape, Polygon):
            return self._intersects_polygon(shape)
        elif isinstance(shape, MultiPolygon):
            return self._intersects_polygon(shape)
        else:
            raise ValueError("Unsupported intersection type %s" % type(shape).__name__)


class PointArray(GeometryFixedArray):
    _element_type = Point

    @property
    def _dtype_class(self):
        return PointDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas MultiPointArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of MultiPoint or
            Point shapes.

        Returns:
            MultiPointArray
        """
        return super().from_geopandas(ga)

    @property
    def length(self):
        return np.zeros(len(self), dtype=np.float64)

    @property
    def area(self):
        return np.zeros(len(self), dtype=np.float64)

    @property
    def x(self):
        res = np.full(len(self), np.nan)
        mask = ~self.isna()
        res[mask] = self.flat_values[0::2][mask]
        return res

    @property
    def y(self):
        res = np.full(len(self), np.nan)
        mask = ~self.isna()
        res[mask] = self.flat_values[1::2][mask]
        return res

    def intersects_bounds(self, bounds, inds=None):
        x0, y0, x1, y1 = bounds
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        xs = self.x
        ys = self.y
        if inds is not None:
            xs = xs[inds]
            ys = ys[inds]

        outside = np.isnan(xs) | (xs < x0) | (xs > x1) | (ys < y0) | (ys > y1)
        return ~outside

    def _intersects_point(self, point, inds):
        flat = self.flat_values
        if inds is None:
            return (flat[0::2] == point.x) & (flat[1::2] == point.y)
        else:
            return (flat[inds * 2] == point.x) & (flat[inds * 2 + 1] == point.y)

    def _intersects_multipoint(self, multipoint, inds):
        flat_points = self.flat_values
        flat_multipoint = multipoint.flat_values
        if inds is None:
            inds = np.arange(len(self))
        return _perform_intersects_multipoint(flat_points, flat_multipoint, inds)

    def _intersects_line(self, line, inds):
        if inds is None:
            inds = np.arange(len(self))
        return _perform_intersects_line(
            self.flat_values, line.buffer_values,  line.buffer_inner_offsets, inds
        )

    def _intersects_polygon(self, polygon, inds):
        if inds is None:
            inds = np.arange(len(self))
        return _perform_intersects_polygon(
            self.flat_values, polygon.buffer_values,  polygon.buffer_inner_offsets, inds
        )

    def intersects(self, shape, inds=None):
        from . import Line, MultiLine, MultiPoint, MultiPolygon, Polygon
        if isinstance(shape, Point):
            return self._intersects_point(shape, inds)
        elif isinstance(shape, MultiPoint):
            return self._intersects_multipoint(shape, inds)
        elif isinstance(shape, Line):
            return self._intersects_line(shape, inds)
        elif isinstance(shape, MultiLine):
            return self._intersects_line(shape, inds)
        elif isinstance(shape, Polygon):
            return self._intersects_polygon(shape, inds)
        elif isinstance(shape, MultiPolygon):
            return self._intersects_polygon(shape, inds)
        else:
            raise ValueError("Unsupported intersection type %s" % type(shape).__name__)


@ngpjit
def _perform_intersects_multipoint(flat_points, flat_multipoint, inds):
    n = len(inds)
    multi_xs = flat_multipoint[0::2]
    multi_ys = flat_multipoint[1::2]
    result = np.zeros(n, dtype=np.bool_)
    for i, j in enumerate(inds):
        x = flat_points[2 * j]
        y = flat_points[2 * j + 1]
        result[i] = np.any((multi_xs == x) & (multi_ys == y))

    return result


@ngpjit
def _perform_intersects_line(flat_points, flat_lines, offsets, inds):
    n = len(inds)
    result = np.zeros(n, dtype=np.bool_)
    for i, j in enumerate(inds):
        x = flat_points[2 * j]
        y = flat_points[2 * j + 1]

        for k in range(len(offsets) - 1):
            flat_line = flat_lines[offsets[k]:offsets[k + 1]]
            line_xs = flat_line[0::2]
            line_ys = flat_line[1::2]
            bounds = (min(line_xs), min(line_ys), max(line_xs), max(line_ys))

            # Check bounding box
            if x < bounds[0] or y < bounds[1] or x > bounds[2] or y > bounds[3]:
                continue

            # Check line vertices
            intersects_vert = np.any((line_xs == x) & (line_ys == y))
            if intersects_vert:
                result[i] = True
                continue

            # Check whether point is on a line segment
            for m in range(len(line_xs) - 1):
                ax0 = line_xs[m]
                ay0 = line_ys[m]
                ax1 = line_xs[m + 1]
                ay1 = line_ys[m + 1]
                intersects_segment = segment_intersects_point(ax0, ay0, ax1, ay1, x, y)
                if intersects_segment:
                    result[i] = True
                    break

    return result


@ngpjit
def _perform_intersects_polygon(flat_points, flat_polygons, offsets, inds):
    n = len(inds)
    result = np.zeros(n, dtype=np.bool_)
    for i, j in enumerate(inds):
        x = flat_points[2 * j]
        y = flat_points[2 * j + 1]

        result[i] = point_intersects_polygon(
            x, y, flat_polygons, offsets
        )
    return result


def _points_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return PointArray([
        [1, 0],
        [1, 2]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(PointDtype)(_points_array_non_empty)
