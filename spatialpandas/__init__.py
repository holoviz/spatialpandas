from importlib.util import find_spec

from . import geometry, spatialindex, tools
from .__version import __version__
from .geodataframe import GeoDataFrame
from .geoseries import GeoSeries
from .tools.sjoin import sjoin

if find_spec("dask"):
    # Import to trigger registration of types with Dask
    import spatialpandas.dask  # noqa

del find_spec

__all__ = [
    "GeoDataFrame",
    "GeoSeries",
    "__version__",
    "geometry",
    "sjoin",
    "spatialindex",
    "tools",
]
