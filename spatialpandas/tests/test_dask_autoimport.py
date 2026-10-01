import dask.dataframe as dd
import pandas as pd
from dask.base import tokenize

import spatialpandas as sp


def test_dask_registration():
    ddf = dd.from_pandas(sp.GeoDataFrame({
        'geom': pd.array(
            [[0, 0], [0, 1, 1, 1], [0, 2, 1, 2, 2, 2]], dtype='MultiPoint[float64]'),
        'v': [1, 2, 3]
    }), npartitions=3)
    assert isinstance(ddf, sp.dask.DaskGeoDataFrame)


def test_tokenize_geometry_array_does_not_iterate(monkeypatch):
    arr = sp.geometry.PointArray([[0, 0], [1, 1], [2, 2]])

    def fail(self):
        raise AssertionError("Tokenizing should not iterate over the elements")

    monkeypatch.setattr(sp.geometry.PointArray, "__iter__", fail)
    tokenize(arr)


def test_tokenize_geometry_array():
    arr = sp.geometry.PointArray([[0, 0], [1, 1], [2, 2], [0, 0]])
    same = sp.geometry.PointArray([[0, 0], [1, 1], [2, 2], [0, 0]])
    multi = sp.geometry.MultiPointArray([[0, 0], [1, 1], [2, 2], [0, 0]])

    assert tokenize(arr) == tokenize(same)
    assert tokenize(arr) != tokenize(multi)
    assert tokenize(arr[:2]) != tokenize(arr[1:3])
    assert tokenize(arr[:2]) != tokenize(arr[:3])
