import copy
import json
import os

import pandas as pd
from dask.dataframe import to_parquet as dd_to_parquet, read_parquet as dd_read_parquet

from pandas.io.parquet import (
    to_parquet as pd_to_parquet, read_parquet as pd_read_parquet
)
import pyarrow as pa
from pyarrow import parquet as pq

from spatialpandas import GeoDataFrame
from spatialpandas.dask import DaskGeoDataFrame
from spatialpandas.geometry.base import to_geometry_array
from spatialpandas.geometry import (
    PointDtype, MultiPointDtype, RingDtype, LineDtype,
    MultiLineDtype, PolygonDtype, MultiPolygonDtype, GeometryDtype
)

_geometry_dtypes = [
    PointDtype, MultiPointDtype, RingDtype, LineDtype,
    MultiLineDtype, PolygonDtype, MultiPolygonDtype
]


def _import_geometry_columns(df, geom_cols):
    new_cols = {}
    for col, type_str in geom_cols.items():
        if col in df and not isinstance(df.dtypes[col], GeometryDtype):
            new_cols[col] = to_geometry_array(df[col], dtype=type_str)

    return df.assign(**new_cols)


def _load_parquet_pandas_metadata(path):
    if not os.path.exists(path):
        raise ValueError("Path not found: " + path)

    if os.path.isdir(path):
        pqds = pa.parquet.ParquetDataset(path)
        metadata = pqds.common_metadata.metadata
    else:
        pf = pa.parquet.ParquetFile(path)
        metadata = pf.metadata.metadata

    return json.loads(
        metadata.get(b'pandas', b'{}').decode('utf')
    )


def _get_geometry_columns(pandas_metadata):
    columns = pandas_metadata.get('columns', [])
    geom_cols = {}
    for col in columns:
        type_string = col.get('numpy_type', None)
        is_geom_col = False
        for geom_type in _geometry_dtypes:
            try:
                geom_type.construct_from_string(type_string)
                is_geom_col = True
            except TypeError:
                pass
        if is_geom_col:
            geom_cols[col["name"]] = col["numpy_type"]

    return geom_cols


def to_parquet(
    df,
    fname,
    compression="snappy",
    index=None,
    **kwargs
):
    # Standard pandas to_parquet with pyarrow engine
    pd_to_parquet(
        df, fname, engine="pyarrow", compression=compression, index=index, **kwargs
    )


def read_parquet(path, columns=None):
    # Load using standard pandas read_parquet
    result = pd_read_parquet(path, engine="auto", columns=columns)

    # Import geometry columns, not needed for pyarrow >= 0.16
    metadata = _load_parquet_pandas_metadata(path)
    geom_cols = _get_geometry_columns(metadata)
    if geom_cols:
        result = _import_geometry_columns(result, geom_cols)

    # Return result
    return GeoDataFrame(result)


def to_parquet_dask(ddf, path, compression="default", storage_options=None, **kwargs):
    assert isinstance(ddf, DaskGeoDataFrame)

    dd_to_parquet(
        ddf, path, engine="pyarrow", compression=compression,
        storage_options=storage_options, **kwargs
    )

    # Write partition bounding boxes to the _metadata file
    partition_bounds = {}
    for series_name in ddf.columns:
        series = ddf[series_name]
        if isinstance(series.dtype, GeometryDtype):
            partition_bounds[series_name] = series.partition_bounds.to_dict()

    spatial_metadata = {'partition_bounds': partition_bounds}
    b_spatial_metadata = json.dumps(spatial_metadata).encode('utf')

    pqds = pq.ParquetDataset(path)
    all_metadata = copy.copy(pqds.common_metadata.metadata)
    all_metadata[b'spatialpandas'] = b_spatial_metadata
    schema = pqds.common_metadata.schema.to_arrow_schema()
    new_schema = schema.with_metadata(all_metadata)
    pq.write_metadata(new_schema, pqds.common_metadata_path)


def read_parquet_dask(path, columns=None, categories=None, storage_options=None, **kwargs):
    result = dd_read_parquet(
        path,
        columns=columns,
        categories=categories,
        storage_options=storage_options,
        engine="pyarrow",
        **kwargs
    )

    # Import geometry columns, not needed for pyarrow >= 0.16
    metadata = _load_parquet_pandas_metadata(path)
    geom_cols = _get_geometry_columns(metadata)
    if not geom_cols:
        # No geometry columns found, regular DaskDataFrame
        return result

    # Convert Dask DataFrame to DaskGeoDataFrame and the partitions and metadata
    # to GeoDataFrames
    result = result.map_partitions(
        lambda df: GeoDataFrame(_import_geometry_columns(df, geom_cols)),
    )

    result = DaskGeoDataFrame(
        result.dask,
        result._name,
        GeoDataFrame(_import_geometry_columns(result._meta, geom_cols)),
        result.divisions,
    )
    # Load bounding box info from _metadata
    pqds = pq.ParquetDataset(path)
    if b'spatialpandas' in pqds.common_metadata.metadata:
        spatial_metadata = json.loads(
            pqds.common_metadata.metadata[b'spatialpandas'].decode('utf')
        )
        if "partition_bounds" in spatial_metadata:
            partition_bounds = {}
            for name in spatial_metadata['partition_bounds']:
                bounds_df = pd.DataFrame(
                    spatial_metadata['partition_bounds'][name]
                )

                # Index labels will be read in as strings.
                # Here we convert to integers, sort by index, then drop index just in
                # case the rows got shuffled on read
                bounds_df = (bounds_df
                             .set_index(bounds_df.index.astype('int'))
                             .sort_index()
                             .reset_index(drop=True))
                bounds_df.index.name = 'partition'

                partition_bounds[name] = bounds_df
            result._partition_bounds = partition_bounds
    return result
