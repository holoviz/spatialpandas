from . import geometry
from . import spatialindex
from .geoseries import GeoSeries
from .geodataframe import GeoDataFrame
try:
    import dask.dataframe
    # Import to trigger registration of types with Dask
    import spatialpandas.dask
except ImportError:
    # Dask dataframe not available
    pass
