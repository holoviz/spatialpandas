import json
from functools import reduce
from glob import has_magic
from numbers import Number
from packaging.version import Version
from pathlib import Path
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
from pyarrow.parquet import ParquetDataset, ParquetFile, read_metadata

from .. import GeoDataFrame
from ..dask import DaskGeoDataFrame
from ..geometry import GeometryDtype
from ..io.utils import (
    PathType,
    _maybe_prepend_protocol,
    validate_coerce_filesystem,
)

# improve pandas compatibility, based on geopandas _compat.py
PANDAS_GE_12 = Version(pd.__version__) >= Version("1.2.0")

# When we drop support for pyarrow < 5 all code related to this can be removed.
LEGACY_PYARROW = Version(pa.__version__) < Version("5.0.0")


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
        if LEGACY_PYARROW:
            basic_kwargs = dict(validate_schema=False)
        else:
            basic_kwargs = dict(use_legacy_dataset=False)

        pqds = ParquetDataset(
            path,
            filesystem=filesystem,
            **basic_kwargs,
            **engine_kwargs,
        )

        if LEGACY_PYARROW:
            common_metadata = pqds.common_metadata
            if common_metadata is None:
                # Get metadata for first piece
                piece = pqds.pieces[0]
                metadata = piece.get_metadata().metadata
            else:
                metadata = pqds.common_metadata.metadata
        else:
            filename = "/".join([_get_parent_path(pqds.files[0]), "_common_metadata"])
            try:
                common_metadata = _read_metadata(filename, filesystem=filesystem)
            except FileNotFoundError:
                # Common metadata doesn't exist, so get metadata for first piece instead
                filename = pqds.files[0]
                common_metadata = _read_metadata(filename, filesystem=filesystem)
            metadata = common_metadata.metadata
    else:
        with filesystem.open(path) as f:
            pf = ParquetFile(f)
        metadata = pf.metadata.metadata

    return json.loads(
        metadata.get(b'pandas', b'{}').decode('utf')
    )


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

    if LEGACY_PYARROW:
        basic_kwargs = dict(validate_schema=False)
    else:
        basic_kwargs = dict(use_legacy_dataset=False)

    # Load using pyarrow to handle parquet files and directories across filesystems
    dataset = ParquetDataset(
        path,
        filesystem=filesystem,
        **basic_kwargs,
        **engine_kwargs,
        **kwargs,
    )

    if LEGACY_PYARROW:
        metadata = _load_parquet_pandas_metadata(
            path,
            filesystem=filesystem,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )
    else:
        metadata = dataset.schema.pandas_metadata

    # If columns specified, prepend index columns to it
    if columns is not None:
        all_columns = set(column['name'] for column in metadata.get('columns', []))
        index_col_metadata = metadata.get('index_columns', [])
        extra_index_columns = []
        for idx_metadata in index_col_metadata:
            if isinstance(idx_metadata, str):
                name = idx_metadata
            elif isinstance(idx_metadata, dict):
                name = idx_metadata.get('name', None)
            else:
                name = None
            if name is not None and name not in columns and name in all_columns:
                extra_index_columns.append(name)

        columns = extra_index_columns + list(columns)

    df = dataset.read(columns=columns).to_pandas()

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

    # Determine partition bounding boxes to save to _metadata file
    partition_bounds = {}
    for series_name in ddf.columns:
        series = ddf[series_name]
        if isinstance(series.dtype, GeometryDtype):
            partition_bounds[series_name] = series.partition_bounds.to_dict()

    spatial_metadata = {'partition_bounds': partition_bounds}
    b_spatial_metadata = json.dumps(spatial_metadata).encode('utf')

    dd_to_parquet(
        ddf,
        path,
        engine="pyarrow",
        compression=compression,
        storage_options=storage_options,
        custom_metadata={b'spatialpandas': b_spatial_metadata},
        write_metadata_file=True,
        **engine_kwargs,
        **kwargs,
    )


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
    if isinstance(path, (str, Path)):
        paths = [path]
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

    files = []
    for p in paths:
        if hasattr(p, "as_posix"):
            p = p.as_posix()
        if has_magic(p):
            # Expand glob paths
            _files = _expand_path(p, filesystem)
            files.extend(_files)
        elif not filesystem.exists(p):
            raise FileNotFoundError(p)
        else:
            files.append(p)

    # Perform read parquet
    result = _perform_read_parquet_dask(
        files,
        columns=columns,
        filesystem=filesystem,
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


def _expand_path(paths, filesystem):
    # Expand glob paths
    files = filesystem.expand_path(paths)
    if isinstance(files, str):
        files = [files]
    # Filter out metadata files
    files = [_ for _ in files if "_metadata" not in _.split("/")[-1]]
    files = _maybe_prepend_protocol(files, filesystem)
    return files


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
    if LEGACY_PYARROW:
        basic_kwargs = dict(validate_schema=False)
    else:
        basic_kwargs = dict(use_legacy_dataset=False)

    datasets = []
    for path in paths:
        if filesystem.isdir(path):
            path = path.rstrip('/')
            path = f"{path}/**/*.parquet"
        if has_magic(path):
            path = _expand_path(path, filesystem)
        d = ParquetDataset(
            path,
            filesystem=filesystem,
            **basic_kwargs,
            **engine_kwargs,
        )
        datasets.append(d)

    # Create delayed partition for each piece
    pieces = []
    for dataset in datasets:
        # Perform natural sort on pieces so that "part.10" comes after "part.2"
        fragments = getattr(dataset, "fragments", None)
        if fragments is None:
            fragments = dataset.pieces
        dataset_pieces = sorted(fragments, key=lambda piece: natural_sort_key(piece.path))
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
    partition_bounds_list = [
        _load_partition_bounds(dataset, filesystem=filesystem)
        for dataset in datasets
    ]
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

    if LEGACY_PYARROW:
        files = paths
    else:
        files = getattr(datasets[0], "files", paths)

    meta = dd_read_parquet(
        files[0],
        columns=cols_no_index,
        filesystem=filesystem,
        engine='pyarrow',
        categories=categories,
        ignore_metadata_file=True,
        **engine_kwargs,
    )._meta

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


def _get_parent_path(path):
    if isinstance(path, Iterable) and not isinstance(path, str):
        path = path[0]
    parent = str(path).rsplit("/", 1)[0]
    return parent if parent else "/"


def _read_metadata(filename, filesystem):
    with filesystem.open(filename, "rb") as f:
        common_metadata = read_metadata(f)
    return common_metadata


def _load_partition_bounds(pqds, filesystem=None):
    partition_bounds = None

    if LEGACY_PYARROW:
        common_metadata = pqds.common_metadata
    else:
        filename = "/".join([_get_parent_path(pqds.files[0]), "_common_metadata"])
        try:
            common_metadata = _read_metadata(filename, filesystem=filesystem)
        except FileNotFoundError:
            common_metadata = None

    if common_metadata is not None and b'spatialpandas' in common_metadata.metadata:
        spatial_metadata = json.loads(
            common_metadata.metadata[b'spatialpandas'].decode('utf')
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
