from hypothesis import given, settings
import dask.dataframe as dd
import pandas as pd
from spatialpandas import GeoSeries, GeoDataFrame
from spatialpandas.dask import DaskGeoDataFrame
from tests.geometry.strategies import (
    st_multipoint_array, st_multiline_array,
    st_point_array)
import numpy as np
from spatialpandas.io import (
    to_parquet, read_parquet, read_parquet_dask, to_parquet_dask
)

hyp_settings = settings(deadline=None, max_examples=100)


@given(
    gp_point=st_point_array(min_size=1, geoseries=True),
    gp_multipoint=st_multipoint_array(min_size=1, geoseries=True),
    gp_multiline=st_multiline_array(min_size=1, geoseries=True),
)
@hyp_settings
def test_parquet(gp_point, gp_multipoint, gp_multiline, tmp_path):
    # Build dataframe
    n = min(len(gp_multipoint), len(gp_multiline))
    df = GeoDataFrame({
        'point': GeoSeries(gp_point[:n]),
        'multipoint': GeoSeries(gp_multipoint[:n]),
        'multiline': GeoSeries(gp_multiline[:n]),
        'a': list(range(n))
    })

    path = tmp_path / 'df.parq'
    to_parquet(df, path)
    df_read = read_parquet(path)
    assert isinstance(df_read, GeoDataFrame)
    assert all(df == df_read)


@given(
    gp_multipoint=st_multipoint_array(min_size=1, geoseries=True),
    gp_multiline=st_multiline_array(min_size=1, geoseries=True),
)
@hyp_settings
def test_parquet_dask(gp_multipoint, gp_multiline, tmp_path):
    # Build dataframe
    n = min(len(gp_multipoint), len(gp_multiline))
    df = GeoDataFrame({
        'points': GeoSeries(gp_multipoint[:n]),
        'lines': GeoSeries(gp_multiline[:n]),
        'a': list(range(n))
    })
    ddf = dd.from_pandas(df, npartitions=3)

    path = tmp_path / 'ddf.parq'
    ddf.to_parquet(path)
    ddf_read = read_parquet_dask(path)

    # Check type
    assert isinstance(ddf_read, DaskGeoDataFrame)

    # Check that partition bounds were loaded
    assert set(ddf_read._partition_bounds) == {'points', 'lines'}
    pd.testing.assert_frame_equal(
        ddf['points'].partition_bounds,
        ddf_read._partition_bounds['points'],
    )
    pd.testing.assert_frame_equal(
        ddf['lines'].partition_bounds,
        ddf_read._partition_bounds['lines'],
    )


@given(
    gp_multipoint=st_multipoint_array(min_size=10, max_size=40, geoseries=True),
    gp_multiline=st_multiline_array(min_size=10, max_size=40, geoseries=True),
)
@hyp_settings
def test_pack_partitions(gp_multipoint, gp_multiline):
    # Build dataframe
    n = min(len(gp_multipoint), len(gp_multiline))
    df = GeoDataFrame({
        'points': GeoSeries(gp_multipoint[:n]),
        'lines': GeoSeries(gp_multiline[:n]),
        'a': list(range(n))
    }).set_geometry('lines')
    ddf = dd.from_pandas(df, npartitions=3)

    # Pack partitions
    ddf_packed = ddf.pack_partitions(npartitions=4)

    # Check the number of partitions
    assert ddf_packed.npartitions == 4

    # Check that rows are now sorted in order of hilbert distance
    total_bounds = df.lines.total_bounds
    hilbert_distances = ddf_packed.lines.map_partitions(
        lambda s: s.hilbert_distance(total_bounds=total_bounds)
    ).compute().values

    # Compute expected total_bounds
    expected_distances = np.sort(
        df.lines.hilbert_distance(total_bounds=total_bounds).values
    )

    np.testing.assert_equal(expected_distances, hilbert_distances)
