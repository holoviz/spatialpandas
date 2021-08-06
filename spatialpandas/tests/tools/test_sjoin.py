import dask.dataframe as dd
import geopandas as gp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given

import spatialpandas as sp
from ..geometry.strategies import st_point_array, st_polygon_array
from ..test_parquet import hyp_settings
from spatialpandas import GeoDataFrame
from spatialpandas.dask import DaskGeoDataFrame

try:
    from geopandas._compat import HAS_RTREE, USE_PYGEOS
    gpd_spatialindex = USE_PYGEOS or HAS_RTREE
except ImportError:
    try:
        import rtree  # noqa
        gpd_spatialindex = rtree
    except Exception:
        gpd_spatialindex = False

if not gpd_spatialindex:
    pytest.skip('Geopandas spatialindex not available to compare against',
                allow_module_level=True)


@pytest.mark.slow
@pytest.mark.parametrize("how", ["inner", "left", "right"])
@given(
    st_point_array(min_size=1, geoseries=True),
    st_polygon_array(min_size=1, geoseries=True),
)
@hyp_settings
def test_sjoin(how, gp_points, gp_polygons):
    # join witgh geopandas
    left_gpdf = gp.GeoDataFrame({
        'geometry': gp_points,
        'a': np.arange(10, 10 + len(gp_points)),
        'c': np.arange(20, 20 + len(gp_points)),
        'v': np.arange(30, 30 + len(gp_points)),
    }).set_index('a')

    right_gpdf = gp.GeoDataFrame({
        'geometry': gp_polygons,
        'a': np.arange(10, 10 + len(gp_polygons)),
        'b': np.arange(20, 20 + len(gp_polygons)),
        'v': np.arange(30, 30 + len(gp_polygons)),
    }).set_index('b')

    # Generate expected result as geopandas GeoDataFrame
    gp_expected = gp.sjoin(left_gpdf, right_gpdf, how=how)
    gp_expected = gp_expected.rename(columns={"v_x": "v_left", "v_y": "v_right"})
    if how == "right":
        gp_expected.index.name = right_gpdf.index.name
    else:
        gp_expected.index.name = left_gpdf.index.name

    # join with spatialpandas
    left_spdf = GeoDataFrame(left_gpdf)
    right_spdf = GeoDataFrame(right_gpdf)

    result = sp.sjoin(
        left_spdf, right_spdf, how=how
    ).sort_values(['v_left', 'v_right'])
    assert isinstance(result, GeoDataFrame)

    # Check pandas results
    if len(gp_expected) == 0:
        assert len(result) == 0
    else:
        expected = GeoDataFrame(gp_expected).sort_values(['v_left', 'v_right'])
        pd.testing.assert_frame_equal(expected, result)

        # left_df as Dask frame
        left_spddf = dd.from_pandas(left_spdf, npartitions=4)

        if how == "right":
            # Right join not supported when left_df is a Dask DataFrame
            with pytest.raises(ValueError):
                sp.sjoin(left_spddf, right_spdf, how=how)
            return
        else:
            result_ddf = sp.sjoin(
                left_spddf, right_spdf, how=how
            )
        assert isinstance(result_ddf, DaskGeoDataFrame)
        assert result_ddf.npartitions <= 4
        result = result_ddf.compute().sort_values(['v_left', 'v_right'])
        pd.testing.assert_frame_equal(expected, result)
