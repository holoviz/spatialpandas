from collections import OrderedDict

import dask
import dask.dataframe as dd
import geopandas as gp
import pandas as pd
import pytest
import shapely.geometry as sg

import spatialpandas as sp
from spatialpandas import GeoDataFrame, GeoSeries

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


def test_import_geopandas_with_none():
    # Construct geopandas dataframe with column for each geometry type that includes
    # Nones
    gp_points = gp.array.from_shapely([
        sg.Point([0, 0]), sg.Point(1, 1), None
    ])

    gp_lines = gp.array.from_shapely([
        sg.LineString([(0, 0), (1, 1)]), None, sg.LineString([(1, 1), (2, 2)])
    ])

    gp_rings = gp.array.from_shapely([
        sg.LineString([(0, 0), (1, 1), (0, 0)]),
        None,
        sg.LineString([(1, 1), (2, 2), (1, 1)]),
    ])

    gp_multilines = gp.array.from_shapely([
        None,
        sg.MultiLineString([[(0, 0), (1, 1)]]),
        sg.MultiLineString([[(1, 1), (2, 2)]])
    ])

    gp_polygons = gp.array.from_shapely([
        None,
        sg.Polygon([(0, 0), (1, 1), (0, 1)]),
        sg.Polygon([(1, 1), (2, 2), (1, 0)])
    ])

    gp_multipolygons = gp.array.from_shapely([
        sg.MultiPolygon([sg.Polygon([(0, 0), (1, 1), (0, 1)])]),
        sg.MultiPolygon([sg.Polygon([(1, 1), (2, 2), (1, 0)])]),
        None
    ])

    gpdf = gp.GeoDataFrame({
        'a': [1, 2, 3],
        'point': gp_points,
        'b': [3, 4, 5],
        'line': gp_lines,
        'ring': gp_rings,
        'multiline': gp_multilines,
        'polygon': gp_polygons,
        'multipolygon': gp_multipolygons,

    })

    # Construct spatialpandas GeoDataFrame
    spdf = sp.GeoDataFrame(gpdf)

    # Check that expected entries are NA
    pd.testing.assert_frame_equal(
        spdf.isna(),
        pd.DataFrame({
            'a': [0, 0, 0],
            'point': [0, 0, 1],
            'b': [0, 0, 0],
            'line': [0, 1, 0],
            'ring': [0, 1, 0],
            'multiline': [1, 0, 0],
            'polygon': [1, 0, 0],
            'multipolygon': [0, 0, 1]
        }, dtype='bool')
    )


def test_drop_geometry_column():
    gp_points = gp.array.from_shapely([
        sg.Point([0, 0]), sg.Point(1, 1)
    ])

    gp_lines = gp.array.from_shapely([
        sg.LineString([(0, 0), (1, 1)]), sg.LineString([(1, 1), (2, 2)])
    ])

    gpdf = gp.GeoDataFrame({
        'a': [1, 2],
        'point': gp_points,
        'b': [3, 4],
        'line1': gp_lines,
        'line2': gp_lines
    }, geometry='line2')

    # Import with no geometry column set
    spdf = sp.GeoDataFrame(gpdf)
    assert spdf.geometry.name == 'line2'

    # Active geometry column preserved when dropping a different non-geometry column
    df = spdf.drop('b', axis=1)
    assert df.geometry.name == 'line2'

    # Active geometry column preserved when dropping a different geometry column
    df = spdf.drop('line1', axis=1)
    assert df.geometry.name == 'line2'

    # Dropping active geometry column results in GeoDataFrame with no active geometry
    df = spdf.drop(['point', 'line2'], axis=1)
    assert df._has_valid_geometry() is False

    # Dropping all geometry columns results in pandas DataFrame
    pdf = spdf.drop(['point', 'line1', 'line2'], axis=1)
    assert type(pdf) is pd.DataFrame
