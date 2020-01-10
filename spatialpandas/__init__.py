from . import geometry, spatialindex, tools
from .geoseries import GeoSeries
from .geodataframe import GeoDataFrame
from .tools.sjoin import sjoin
import param as _param

try:
    import dask.dataframe
    # Import to trigger registration of types with Dask
    import spatialpandas.dask
except ImportError:
    # Dask dataframe not available
    pass

__version__ = str(_param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="spatialpandas")
)
