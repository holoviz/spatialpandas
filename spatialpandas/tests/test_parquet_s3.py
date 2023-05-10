import logging
import os
import shlex
import subprocess
import time

import dask.dataframe as dd
import numpy as np
import pytest

from spatialpandas import GeoDataFrame, GeoSeries
from spatialpandas.geometry import PointArray
from spatialpandas.io import read_parquet, read_parquet_dask, to_parquet, to_parquet_dask


pytest.importorskip("moto")
geopandas = pytest.importorskip("geopandas")
s3fs = pytest.importorskip("s3fs")
requests = pytest.importorskip("requests")

logging.getLogger("botocore").setLevel(logging.INFO)


@pytest.fixture(scope="module", autouse=True)
def s3_fixture():
    """Writable local S3 system.

    Taken from `universal_pathlib/upath/tests` and `s3fs/tests/test_s3fs.py`.
    """
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

    port = 5555
    bucket_name = "test_bucket"
    endpoint_url = f"http://127.0.0.1:{port}/"
    proc = subprocess.Popen(
        shlex.split(f"moto_server s3 -p {port}"),
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    try:
        timeout = 5
        while timeout > 0:
            try:
                r = requests.get(endpoint_url, timeout=10)
                if r.ok:
                    break
            except Exception:  # pragma: no cover
                pass
            timeout -= 0.1  # pragma: no cover
            time.sleep(0.1)  # pragma: no cover
        anon = False
        s3so = {
            "anon": anon,
            "endpoint_url": endpoint_url,
        }
        fs = s3fs.S3FileSystem(**s3so)
        fs.mkdir(bucket_name)
        assert fs.exists(bucket_name)
        yield f"s3://{bucket_name}", s3so
    finally:
        proc.terminate()
        proc.wait()


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


class TestS3ParquetDask:
    @staticmethod
    def test_read_parquet_dask_remote_glob_parquet(s3_parquet_dask):
        path, s3so, sdf = s3_parquet_dask
        result = read_parquet_dask(f"{path}/*.parquet", storage_options=s3so).compute()
        assert result.equals(sdf)

    @staticmethod
    def test_read_parquet_dask_remote_glob_all(s3_parquet_dask):
        path, s3so, sdf = s3_parquet_dask
        result = read_parquet_dask(f"{path}/*", storage_options=s3so).compute()
        assert result.equals(sdf)

    @staticmethod
    def test_read_parquet_dask_remote_dir(s3_parquet_dask):
        path, s3so, sdf = s3_parquet_dask
        result = read_parquet_dask(path, storage_options=s3so).compute()
        assert result.equals(sdf)

    @staticmethod
    def test_read_parquet_dask_remote_dir_slash(s3_parquet_dask):
        path, s3so, sdf = s3_parquet_dask
        result = read_parquet_dask(f"{path}/", storage_options=s3so).compute()
        assert result.equals(sdf)


def test_read_parquet_remote(s3_parquet_pandas):
    path, s3so, sdf = s3_parquet_pandas
    result = read_parquet(path, storage_options=s3so)
    assert result.equals(sdf)
