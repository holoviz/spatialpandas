from spatialpandas import GeoDataFrame, GeoSeries
import pandas as pd
from collections import OrderedDict
import pytest
import dask.dataframe as dd
import dask

import spatialpandas as sp
import geopandas as gp
import shapely.geometry as sg

dask.config.set(scheduler="single-threaded")


@pytest.mark.parametrize('use_dask', [False, True])
def test_active_geometry(use_dask):
    gdf = GeoDataFrame(OrderedDict([
        ('a', [3, 2, 1]),
        ('points', pd.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4]], dtype='multipoint')),
        ('line', pd.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4]], dtype='line')),
    ]))

    if use_dask:
        gdf = dd.from_pandas(gdf, npartitions=2)

    # geometry start out as first compatible column in data frame
    assert gdf.geometry.name == 'points'

    # set_geometry default to copy operation
    assert gdf.set_geometry('line').geometry.name == 'line'
    assert gdf.geometry.name == 'points'

    # set_geometry inplace mutates geometry column
    if use_dask:
        # inplace not supported for DaskGeoDataFrame
        gdf = gdf.set_geometry('line')
    else:
        gdf.set_geometry('line', inplace=True)
    assert gdf.geometry.name == 'line'

    # Activate geometry propagates through slicing
    sliced_gdf = gdf.loc[[0, 2, 1, 0]]
    assert isinstance(sliced_gdf, type(gdf))
    assert sliced_gdf.geometry.name == 'line'

    # Select columns not including active geometry
    selected_gdf = gdf[['a', 'points']]
    with pytest.raises(ValueError):
        selected_gdf.geometry

    assert selected_gdf.set_geometry('points').geometry.name == 'points'


def test_dataframe_slice_types():
    gdf = GeoDataFrame({
        'a': [3, 2, 1],
        'b': [10, 11, 12],
        'points': pd.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4]], dtype='multipoint'),
        'line': pd.array([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4]], dtype='line'),
    })

    assert isinstance(gdf['a'], pd.Series)
    assert isinstance(gdf['points'], GeoSeries)
    assert isinstance(gdf[['a', 'b']], pd.DataFrame)
    assert isinstance(gdf[['a', 'line']], GeoDataFrame)


def test_import_geopandas_preserves_geometry_column():
    gp_points = gp.array.from_shapely([
        sg.Point([0, 0]), sg.Point(1, 1)
    ])

    gp_lines = gp.array.from_shapely([
        sg.LineString([(0, 0), (1, 1)]), sg.LineString([(1, 1), (2, 2)])
    ])

    gpdf = gp.GeoDataFrame({
        'a': [1, 2], 'point': gp_points, 'b': [3, 4], 'line': gp_lines
    })

    # Import with no geometry column set
    spdf = sp.GeoDataFrame(gpdf)
    assert spdf.geometry.name == 'point'

    # Import with geometry column set
    for geom_name in ['point', 'line']:
        gpdf = gpdf.set_geometry(geom_name)
        spdf = sp.GeoDataFrame(gpdf)
        assert spdf.geometry.name == geom_name
