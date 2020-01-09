import numpy as np
import pandas as pd

import geopandas as gp
import spatialpandas as sp
from hypothesis import given
from spatialpandas import GeoDataFrame
from tests.geometry.strategies import st_point_array, st_polygon_array
from tests.test_parquet import hyp_settings
import hypothesis.strategies as hs


@given(
    st_point_array(min_size=1, geoseries=True),
    st_polygon_array(min_size=1, geoseries=True),
    hs.sampled_from(["inner", "left", "right"])
)
@hyp_settings
def test_sjoin(gp_points, gp_polygons, how):
    # join with geopandas
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
    right_spds = GeoDataFrame(right_gpdf)

    result = sp.sjoin(
        left_spdf, right_spds, how=how
    ).sort_values(['v_left', 'v_right'])

    # Check results
    if len(result) == 0:
        assert len(gp_expected) == 0
    else:
        expected = GeoDataFrame(gp_expected).sort_values(['v_left', 'v_right'])
        pd.testing.assert_frame_equal(expected, result)
