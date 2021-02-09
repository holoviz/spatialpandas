import dask.dataframe as dd
import pandas as pd

import spatialpandas as sp


def test_dask_registration():
    ddf = dd.from_pandas(sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    }), npartitions=3)
    assert isinstance(ddf, sp.dask.DaskGeoDataFrame)
