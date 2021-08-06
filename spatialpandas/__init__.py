from distutils.version import LooseVersion

import param as _param
import pandas as _pd

from . import geometry, spatialindex, tools  # noqa
from .geodataframe import GeoDataFrame  # noqa
from .geoseries import GeoSeries  # noqa
from .tools.sjoin import sjoin  # noqa

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

# improve pandas compatibility, based on geopandas _compat.py
PANDAS_GE_12 = str(_pd.__version__) >= LooseVersion("1.2.0")