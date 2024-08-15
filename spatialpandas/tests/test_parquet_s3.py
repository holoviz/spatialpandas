import logging
import os

import dask.dataframe as dd
import numpy as np
import pytest

from spatialpandas import GeoDataFrame, GeoSeries
from spatialpandas.geometry import PointArray
from spatialpandas.io import read_parquet, read_parquet_dask, to_parquet, to_parquet_dask

pytest.importorskip("moto")
geopandas = pytest.importorskip("geopandas")
s3fs = pytest.importorskip("s3fs")

logging.getLogger("botocore").setLevel(logging.INFO)

pytestmark = pytest.mark.xdist_group("s3")

PORT = 5555
ENDPOINT_URL = f"http://127.0.0.1:{PORT}/"
BUCKET_NAME = "test_bucket"


@pytest.fixture(scope="module", autouse=True)
def s3_fixture():
    """Writable local S3 system.

    Taken from `universal_pathlib/upath/tests` and `s3fs/tests/test_s3fs.py`.
    """
    from moto.moto_server.threaded_moto_server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=PORT)
    server.start()

    if "BOTO_CONFIG" not in os.environ:  # pragma: no cover
        os.environ["BOTO_CONFIG"] = "/dev/null"
    if "AWS_ACCESS_KEY_ID" not in os.environ:  # pragma: no cover
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    if "AWS_SECURITY_TOKEN" not in os.environ:  # pragma: no cover
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
    if "AWS_SESSION_TOKEN" not in os.environ:  # pragma: no cover
        os.environ["AWS_SESSION_TOKEN"] = "testing"
    if "AWS_DEFAULT_REGION" not in os.environ:  # pragma: no cover
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    s3so = {"anon": False, "endpoint_url": ENDPOINT_URL}
    fs = s3fs.S3FileSystem(**s3so)
    fs.mkdir(BUCKET_NAME)
    assert fs.exists(BUCKET_NAME)
    yield f"s3://{BUCKET_NAME}", s3so
    server.stop()


@pytest.fixture(scope="module")
def sdf():
    src_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
    points = PointArray(src_array)
    return GeoDataFrame({"point": GeoSeries(points)})


@pytest.fixture(scope="module")
def s3_parquet_dask(s3_fixture, sdf):
    path, s3so = s3_fixture
    path = f"{path}/test_dask"
    ddf = dd.from_pandas(sdf, npartitions=2)
    to_parquet_dask(ddf, path, storage_options=s3so)
    fs = s3fs.S3FileSystem(**s3so)
    assert fs.exists(path) and fs.isdir(path)
    yield path, s3so, sdf


@pytest.fixture(scope="module")
def s3_parquet_pandas(s3_fixture, sdf):
    path, s3so = s3_fixture
    path = f"{path}/test_pandas.parquet"
    to_parquet(sdf, path, storage_options=s3so)
    fs = s3fs.S3FileSystem(**s3so)
    assert fs.exists(path) and fs.isfile(path)
    yield path, s3so, sdf


@pytest.mark.parametrize(
    "fpath",
    ["{path}/*.parquet", "{path}/*", "{path}", "{path}/"],
    ids=["glob_parquet", "glob_all", "dir", "dir_slash"],
)
def test_read_parquet_dask_remote(s3_parquet_dask, fpath):
    path, s3so, sdf = s3_parquet_dask
    result = read_parquet_dask(fpath.format(path=path), storage_options=s3so).compute()
    assert result.equals(sdf)


def test_read_parquet_pandas_remote(s3_parquet_pandas):
    path, s3so, sdf = s3_parquet_pandas
    result = read_parquet(path, storage_options=s3so)
    assert result.equals(sdf)
