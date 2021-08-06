import param as _param

from . import geometry, spatialindex, tools
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

__version__ = str(
    _param.version.Version(
        fpath=__file__,
        archive_commit="$Format:%h$",
        reponame="spatialpandas",
    ))

__all__ = [
    "GeoDataFrame",
    "GeoSeries",
    "geometry",
    "sjoin",
    "spatialindex",
    "tools",
]
