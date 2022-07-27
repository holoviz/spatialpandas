# Script to create parquet files that are read in test_parquet.
# The files can be created with particular version of pyarrow and dask, and
# checked that they can be read with different versions of them.

import dask.dataframe as dd
from hypothesis.errors import NonInteractiveExampleWarning
import os
from pyarrow import __version__ as pyarrow_version
from spatialpandas import GeoDataFrame
from spatialpandas.io import to_parquet
from spatialpandas.tests.geometry.strategies import st_multiline_array
import warnings


directory = "test_data"


def _generate_test_geodata():
    n = 5
    warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
    lines = st_multiline_array(min_size=n, max_size=n, geoseries=True).example()
    return lines


def _write(use_dask=False):
    if use_dask:
        filename = f"dask_{pyarrow_version}.parq"
    else:
        filename = f"serial_{pyarrow_version}.parq"

    lines = _generate_test_geodata()
    df = GeoDataFrame({'multiline': lines, 'a': list(range(len(lines)))})

    path = os.path.join(directory, filename)

    if use_dask:
        ddf = dd.from_pandas(df, npartitions=2)
        ddf.to_parquet(path)
    else:
        to_parquet(df, path)


def _write_repartitioned():
    filename = f"dask_repart_{pyarrow_version}.parq"

    lines = _generate_test_geodata()
    df = GeoDataFrame({'multiline': lines, 'a': list(range(len(lines)))})
    ddf = dd.from_pandas(df, npartitions=2)

    path = os.path.join(directory, filename)

    ddf.pack_partitions_to_parquet(path, npartitions=2, overwrite=True)


if __name__ == '__main__':
    _write(False)
    _write(True)
    _write_repartitioned()
