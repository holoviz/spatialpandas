from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.line2d import (
    Line2dDtype, Line2d, Line2dArray
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class Ring2dDtype(Line2dDtype):
    _geometry_name = 'ring2d'

    @classmethod
    def construct_array_type(cls, *args):
        return Ring2dArray


class Ring2d(Line2d):
    def to_shapely(self):
        """
        Convert to shapely shape

        Returns:
            shapely LinearRing shape
        """
        import shapely.geometry as sg
        line_coords = self.data.to_numpy()
        return sg.LinearRing(line_coords.reshape(len(line_coords) // 2, 2))

    @classmethod
    def from_shapely(cls, shape):
        """
        Build a spatialpandas Ring2d object from a shapely shape

        Args:
            shape: A shapely LinearRing shape

        Returns:
            spatialpandas Ring2d
        """
        return super(Ring2d, cls).from_shapely(shape)


class Ring2dArray(Line2dArray):
    _element_type = Line2d

    @property
    def _dtype_class(self):
        return Ring2dDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas Ring2dArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of LinearRing shapes.

        Returns:
            Ring2dArray
        """
        return super(Line2dArray, cls).from_geopandas(ga)


def ring_array_non_empty(dtype):
    return Ring2dArray([
        [0, 0, 1, 0, 1, 1, 0, 0],
        [2, 2, 2, 3, 3, 3, 2, 2]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(Ring2dDtype)(ring_array_non_empty)
