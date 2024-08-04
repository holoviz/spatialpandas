from . import geometry, spatialindex, tools
from .__version import __version__
from .geodataframe import GeoDataFrame
from .geoseries import GeoSeries
from .tools.sjoin import sjoin

try:
    import dask.dataframe  # noqa

    # Import to trigger registration of types with Dask
    import spatialpandas.dask  # noqa
except ImportError:
    # Dask dataframe not available
    pass


__all__ = [
    "GeoDataFrame",
    "GeoSeries",
    "__version__",
    "geometry",
    "sjoin",
    "spatialindex",
    "tools",
]
