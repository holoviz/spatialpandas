from pandas.core.dtypes.dtypes import register_extension_dtype

from spatialpandas.geometry.line import (
    LineDtype, Line, LineArray
)
from dask.dataframe.extensions import make_array_nonempty


@register_extension_dtype
class RingDtype(LineDtype):
    _geometry_name = 'ring'

    @classmethod
    def construct_array_type(cls, *args):
        return RingArray


class Ring(Line):
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
        Build a spatialpandas Ring object from a shapely shape

        Args:
            shape: A shapely LinearRing shape

        Returns:
            spatialpandas Ring
        """
        return super(Ring, cls).from_shapely(shape)


class RingArray(LineArray):
    _element_type = Line

    @property
    def _dtype_class(self):
        return RingDtype

    @classmethod
    def from_geopandas(cls, ga):
        """
        Build a spatialpandas RingArray from a geopandas GeometryArray or
        GeoSeries.

        Args:
            ga: A geopandas GeometryArray or GeoSeries of LinearRing shapes.

        Returns:
            RingArray
        """
        return super(RingArray, cls).from_geopandas(ga)


def _ring_array_non_empty(dtype):
    """
    Create an example length 2 array to register with Dask.
    See https://docs.dask.org/en/latest/dataframe-extend.html#extension-arrays
    """
    return RingArray([
        [0, 0, 1, 0, 1, 1, 0, 0],
        [2, 2, 2, 3, 3, 3, 2, 2]
    ], dtype=dtype)


if make_array_nonempty:
    make_array_nonempty.register(RingDtype)(_ring_array_non_empty)
