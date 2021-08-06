import copy
import json
import pathlib
from distutils.version import LooseVersion
from functools import reduce
from glob import has_magic
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import fsspec
import pandas as pd
import pyarrow as pa
from dask import delayed
from dask.dataframe import from_delayed, from_pandas
from dask.dataframe import read_parquet as dd_read_parquet
from dask.dataframe import to_parquet as dd_to_parquet  # noqa
from dask.utils import natural_sort_key
from pandas.io.parquet import to_parquet as pd_to_parquet
from pyarrow import parquet as pq

from .. import GeoDataFrame
from ..dask import DaskGeoDataFrame
from ..geometry import (GeometryDtype, LineDtype, MultiLineDtype,
                                    MultiPointDtype, MultiPolygonDtype,
                                    PointDtype, PolygonDtype, RingDtype)
from ..geometry.base import to_geometry_array
from ..io.utils import (
    PathType,
    _maybe_prepend_protocol,
    validate_coerce_filesystem,
)

# improve pandas compatibility, based on geopandas _compat.py
PANDAS_GE_12 = str(pd.__version__) >= LooseVersion("1.2.0")

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


def _load_parquet_pandas_metadata(
    path,
    filesystem=None,
    storage_options=None,
    engine_kwargs=None,
):
    engine_kwargs = engine_kwargs or {}
    filesystem = validate_coerce_filesystem(path, filesystem, storage_options)
    if not filesystem.exists(path):
        raise ValueError("Path not found: " + path)

    if filesystem.isdir(path):
        pqds = pq.ParquetDataset(
            path,
            filesystem=filesystem,
            validate_schema=False,
            **engine_kwargs,
        )
        common_metadata = pqds.common_metadata
        if common_metadata is None:
            # Get metadata for first piece
            piece = pqds.pieces[0]
            metadata = piece.get_metadata().metadata
        else:
            metadata = pqds.common_metadata.metadata
    else:
        with filesystem.open(path) as f:
            pf = pq.ParquetFile(f)
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
    df: GeoDataFrame,
    path: PathType,
    compression: Optional[str] = "snappy",
    filesystem: Optional[fsspec.AbstractFileSystem] = None,
    index: Optional[bool] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    if filesystem is not None:
        filesystem = validate_coerce_filesystem(path, filesystem, storage_options)

    # Standard pandas to_parquet with pyarrow engine
    to_parquet_args = {
        "df": df,
        "path": path,
        "engine": "pyarrow",
        "compression": compression,
        "filesystem": filesystem,
        "index": index,
        **kwargs,
    }

    if PANDAS_GE_12:
        to_parquet_args.update({"storage_options": storage_options})
    else:
        if filesystem is None:
            filesystem = validate_coerce_filesystem(path, filesystem, storage_options)
            to_parquet_args.update({"filesystem": filesystem})

    pd_to_parquet(**to_parquet_args)


def read_parquet(
    path: PathType,
    columns: Optional[Iterable[str]] = None,
    filesystem: Optional[fsspec.AbstractFileSystem] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> GeoDataFrame:
    engine_kwargs = engine_kwargs or {}
    filesystem = validate_coerce_filesystem(path, filesystem, storage_options)

    # Load pandas parquet metadata
    metadata = _load_parquet_pandas_metadata(
        path,
        filesystem=filesystem,
        storage_options=storage_options,
        engine_kwargs=engine_kwargs,
    )

    # If columns specified, prepend index columns to it
    if columns is not None:
        index_col_metadata = metadata.get('index_columns', [])
        extra_index_columns = []
        for idx_metadata in index_col_metadata:
            if isinstance(idx_metadata, str):
                name = idx_metadata
            elif isinstance(idx_metadata, dict):
                name = idx_metadata.get('name', None)
            else:
                name = None

            if name is not None and name not in columns:
                extra_index_columns.append(name)

        columns = extra_index_columns + list(columns)

    # Load using pyarrow to handle parquet files and directories across filesystems
    df = pq.ParquetDataset(
        path,
        filesystem=filesystem,
        validate_schema=False,
        **engine_kwargs,
        **kwargs,
    ).read(columns=columns).to_pandas()

    # Import geometry columns, not needed for pyarrow >= 0.16
    geom_cols = _get_geometry_columns(metadata)
    if geom_cols:
        df = _import_geometry_columns(df, geom_cols)

    # Return result
    return GeoDataFrame(df)


def to_parquet_dask(
    ddf: DaskGeoDataFrame,
    path: PathType,
    compression: Optional[str] = "snappy",
    filesystem: Optional[fsspec.AbstractFileSystem] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    engine_kwargs = engine_kwargs or {}
    
    if not isinstance(ddf, DaskGeoDataFrame):
        raise TypeError(f"Expected DaskGeoDataFrame not {type(ddf)}")
    filesystem = validate_coerce_filesystem(path, filesystem, storage_options)
    if kwargs.get("overwrite", False) and \
        not kwargs.get("append", False) and \
            path and filesystem.isdir(path):
        filesystem.rm(path, recursive=True)

    dd_to_parquet(
        ddf,
        path,
        engine="pyarrow",
        compression=compression,
        storage_options=storage_options,
        **kwargs,
    )

    # Write partition bounding boxes to the _metadata file
    partition_bounds = {}
    for series_name in ddf.columns:
        series = ddf[series_name]
        if isinstance(series.dtype, GeometryDtype):
            if series._partition_bounds is None:
                # Bounds are not already computed. Compute bounds from the parquet file
                # that was just written.
                filesystem.invalidate_cache(path)
                series = read_parquet_dask(
                    path,
                    columns=[series_name],
                    filesystem=filesystem,
                    load_divisions=False,
                    storage_options=storage_options,
                )[series_name]
            partition_bounds[series_name] = series.partition_bounds.to_dict()

    spatial_metadata = {'partition_bounds': partition_bounds}
    b_spatial_metadata = json.dumps(spatial_metadata).encode('utf')

    pqds = pq.ParquetDataset(
        path,
        filesystem=filesystem,
        validate_schema=False,
        **engine_kwargs,
    )
    all_metadata = copy.copy(pqds.common_metadata.metadata)
    all_metadata[b'spatialpandas'] = b_spatial_metadata
    schema = pqds.common_metadata.schema.to_arrow_schema()
    new_schema = schema.with_metadata(all_metadata)
    with filesystem.open(pqds.common_metadata_path, 'wb') as f:
        pq.write_metadata(new_schema, f)


def read_parquet_dask(
    path: PathType,
    columns: Optional[Iterable[str]] = None,
    filesystem: Optional[fsspec.AbstractFileSystem] = None,
    load_divisions: Optional[bool] = False,
    geometry: Optional[str] = None,
    bounds: Optional[Tuple[Number, Number, Number, Number]] = None,
    categories: Optional[Union[List[str], Dict[str, str]]] = None,
    build_sindex: Optional[bool] = False,
    storage_options: Optional[Dict[str, Any]] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> DaskGeoDataFrame:
    """Read spatialpandas parquet dataset(s) as DaskGeoDataFrame.

    Datasets are assumed to have been written with the
    DaskGeoDataFrame.to_parquet or
    DaskGeoDataFrame.pack_partitions_to_parquet methods.

    Args:
        path: Path to spatialpandas parquet dataset, or list of paths to datasets, or
            glob string referencing one or more parquet datasets.
        columns: List of columns to load
        filesystem: fsspec filesystem to use to read datasets
        load_divisions: If True, attempt to load the hilbert_distance divisions for
            each partition.  Only available for datasets written using the
            pack_partitions_to_parquet method.
        geometry: The name of the column to use as the geometry column of the
            resulting DaskGeoDataFrame. Defaults to the first geometry column in the
            dataset.
        bounds: If specified, only load partitions that have the potential to intersect
            with the provided bounding box coordinates. bounds is a length-4 tuple of
            the form (xmin, ymin, xmax, ymax).
        categories : list, dict or None
            For any fields listed here, if the parquet encoding is Dictionary,
            the column will be created with dtype category. Use only if it is
            guaranteed that the column is encoded as dictionary in all row-groups.
            If a list, assumes up to 2**16-1 labels; if a dict, specify the number
            of labels expected; if None, will load categories automatically for
            data written by dask/fastparquet, not otherwise.
        build_sindex : boolean
            Whether to build partition level spatial indexes to speed up indexing.
        storage_options: Key/value pairs to be passed on to the file-system backend, if any.
        engine_kwargs: pyarrow.parquet engine-related keyword arguments. 
    Returns:
    DaskGeoDataFrame
    """
    # Normalize path to a list
    if isinstance(path, (str, pathlib.Path)):
        paths = [str(path)]
    else:
        paths = list(path)
        if not paths:
            raise ValueError('Empty path specification')

    # Infer filesystem
    filesystem = validate_coerce_filesystem(
        paths[0],
        filesystem,
        storage_options,
    )

    # Expand glob
    if len(paths) == 1 and has_magic(paths[0]):
        paths = filesystem.glob(paths[0])
        paths = _maybe_prepend_protocol(paths, filesystem)

    # Perform read parquet
    result = _perform_read_parquet_dask(
        paths,
        columns,
        filesystem,
        load_divisions=load_divisions,
        geometry=geometry,
        bounds=bounds,
        categories=categories,
        storage_options=storage_options,
        engine_kwargs=engine_kwargs,
    )

    if build_sindex:
        result = result.build_sindex()

    return result


def _perform_read_parquet_dask(
    paths,
    columns,
    filesystem,
    load_divisions,
    geometry=None,
    bounds=None,
    categories=None,
    storage_options=None,
    engine_kwargs=None,
):
    engine_kwargs = engine_kwargs or {}
    filesystem = validate_coerce_filesystem(
        paths[0],
        filesystem,
        storage_options,
    )
    datasets = [
        pa.parquet.ParquetDataset(
            path,
            filesystem=filesystem,
            validate_schema=False,
            **engine_kwargs,
        ) for path in paths
    ]

    # Create delayed partition for each piece
    pieces = []
    for dataset in datasets:
        # Perform natural sort on pieces so that "part.10" comes after "part.2"
        dataset_pieces = sorted(dataset.pieces, key=lambda piece: natural_sort_key(piece.path))
        pieces.extend(dataset_pieces)

    delayed_partitions = [
        delayed(read_parquet)(
            piece.path,
            columns=columns,
            filesystem=filesystem,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )
        for piece in pieces
    ]

    # Load divisions
    if load_divisions:
        div_mins_list, div_maxes_list = zip(*[
            _load_divisions(dataset) for dataset in datasets
        ])

        div_mins = reduce(lambda a, b: a + b, div_mins_list, [])
        div_maxes = reduce(lambda a, b: a + b, div_maxes_list, [])
    else:
        div_mins = None
        div_maxes = None

    # load partition bounds
    partition_bounds_list = [_load_partition_bounds(dataset) for dataset in datasets]
    if not any([b is None for b in partition_bounds_list]):
        partition_bounds = {}
        # We have partition bounds for all datasets
        for partition_bounds_el in partition_bounds_list:
            for col, col_bounds in partition_bounds_el.items():
                col_bounds_list = partition_bounds.get(col, [])
                col_bounds_list.append(col_bounds)
                partition_bounds[col] = col_bounds_list

        # Concat bounds for each geometry column
        for col in list(partition_bounds):
            partition_bounds[col] = pd.concat(
                partition_bounds[col], axis=0
            ).reset_index(drop=True)
            partition_bounds[col].index.name = 'partition'
    else:
        partition_bounds = {}

    # Use Dask's read_parquet to get metadata
    if columns is not None:
        cols_no_index = [col for col in columns if col != "hilbert_distance"]
    else:
        cols_no_index = None

    meta = dd_read_parquet(
        paths[0],
        columns=cols_no_index,
        filesystem=filesystem,
        engine='pyarrow',
        categories=categories,
        gather_statistics=False,
        storage_options=storage_options,
        **engine_kwargs,
    )._meta

    # Import geometry columns in meta, not needed for pyarrow >= 0.16
    metadata = _load_parquet_pandas_metadata(
        paths[0],
        filesystem=filesystem,
        storage_options=storage_options,
        engine_kwargs=engine_kwargs,
    )
    geom_cols = _get_geometry_columns(metadata)
    if geom_cols:
        meta = _import_geometry_columns(meta, geom_cols)
    meta = GeoDataFrame(meta)

    # Handle geometry in meta
    if geometry:
        meta = meta.set_geometry(geometry)

    geometry = meta.geometry.name

    # Filter partitions by bounding box
    if bounds and geometry in partition_bounds:
        # Unpack bounds coordinates and make sure
        x0, y0, x1, y1 = bounds

        # Make sure x0 < c1
        if x0 > x1:
            x0, x1 = x1, x0

        # Make sure y0 < y1
        if y0 > y1:
            y0, y1 = y1, y0

        # Make DataFrame with bounds and parquet piece
        partitions_df = partition_bounds[geometry].assign(
            delayed_partition=delayed_partitions
        )

        if load_divisions:
            partitions_df = partitions_df.assign(div_mins=div_mins, div_maxes=div_maxes)

        inds = ~(
            (partitions_df.x1 < x0) |
            (partitions_df.y1 < y0) |
            (partitions_df.x0 > x1) |
            (partitions_df.y0 > y1)
        )

        partitions_df = partitions_df[inds]
        for col in list(partition_bounds):
            partition_bounds[col] = partition_bounds[col][inds]
            partition_bounds[col].reset_index(drop=True, inplace=True)
            partition_bounds[col].index.name = "partition"

        delayed_partitions = partitions_df.delayed_partition.tolist()
        if load_divisions:
            div_mins = partitions_df.div_mins
            div_maxes = partitions_df.div_maxes

    if load_divisions:
        divisions = div_mins + [div_maxes[-1]]
        if divisions != sorted(divisions):
            raise ValueError(
                "Cannot load divisions because the discovered divisions are unsorted.\n"
                "Set load_divisions=False to skip loading divisions."
            )
    else:
        divisions = None

    # Create DaskGeoDataFrame
    if delayed_partitions:
        result = from_delayed(
            delayed_partitions, divisions=divisions, meta=meta, verify_meta=False
        )
    else:
        # Single partition empty result
        result = from_pandas(meta, npartitions=1)

    # Set partition bounds
    if partition_bounds:
        result._partition_bounds = partition_bounds

    return result


def _load_partition_bounds(pqds):
    partition_bounds = None
    if (pqds.common_metadata is not None and
            b'spatialpandas' in pqds.common_metadata.metadata):
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

    return partition_bounds


def _load_divisions(pqds):
    fmd = pqds.metadata
    row_groups = [fmd.row_group(i) for i in range(fmd.num_row_groups)]
    rg0 = row_groups[0]
    div_col = None
    for c in range(rg0.num_columns):
        if rg0.column(c).path_in_schema == "hilbert_distance":
            div_col = c
            break

    if div_col is None:
        # No hilbert_distance column found
        raise ValueError(
            "Cannot load divisions because hilbert_distance index not found"
        )

    mins, maxes = zip(*[
        (rg.column(div_col).statistics.min, rg.column(div_col).statistics.max)
        for rg in row_groups
    ])

    return list(mins), list(maxes)
