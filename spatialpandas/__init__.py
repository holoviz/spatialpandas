from . import geometry
from . import spatialindex
from .geoseries import GeoSeries
from .geodataframe import GeoDataFrame
import param
try:
    import dask.dataframe
    # Import to trigger registration of types with Dask
    import spatialpandas.dask
except ImportError:
    # Dask dataframe not available
    pass

__version__ = str(param.version.Version(
    fpath=__file__, archive_commit="$Format:%h$", reponame="spatialpandas")
)
