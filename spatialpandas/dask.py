import copy
import json
import os
import uuid
from inspect import signature

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from retrying import retry

import dask
import dask.dataframe as dd
from dask import delayed
from dask.dataframe.core import get_parallel_type
from dask.dataframe.partitionquantiles import partition_quantiles
from dask.dataframe.extensions import make_array_nonempty
try:
    from dask.dataframe.dispatch import make_meta_dispatch
    from dask.dataframe.backends import meta_nonempty
except ImportError:
    from dask.dataframe.utils import make_meta as make_meta_dispatch, meta_nonempty

from .geodataframe import GeoDataFrame
from .geometry.base import GeometryDtype, _BaseCoordinateIndexer
from .geoseries import GeoSeries
from .spatialindex import HilbertRtree


class DaskGeoSeries(dd.Series):
    def __init__(self, dsk, name, meta, divisions, *args, **kwargs):
        super().__init__(dsk, name, meta, divisions)

        # Init backing properties
        self._partition_bounds = None
        self._partition_sindex = None

    @property
    def bounds(self):
        return self.map_partitions(lambda s: s.bounds)

    @property
    def total_bounds(self):
        partition_bounds = self.partition_bounds
        return (
            np.nanmin(partition_bounds['x0']),
            np.nanmin(partition_bounds['y0']),
            np.nanmax(partition_bounds['x1']),
            np.nanmax(partition_bounds['y1']),
        )

    @property
    def partition_bounds(self):
        if self._partition_bounds is None:
            self._partition_bounds = self.map_partitions(
                lambda s: pd.DataFrame(
                    [s.total_bounds], columns=['x0', 'y0', 'x1', 'y1']
                )
            ).compute().reset_index(drop=True)
            self._partition_bounds.index.name = 'partition'
        return self._partition_bounds

    @property
    def area(self):
        return self.map_partitions(lambda s: s.area)

    @property
    def length(self):
        return self.map_partitions(lambda s: s.length)

    @property
    def partition_sindex(self):
        if self._partition_sindex is None:
            self._partition_sindex = HilbertRtree(self.partition_bounds.values)
        return self._partition_sindex

    @property
    def cx(self):
        return _DaskCoordinateIndexer(self, self.partition_sindex)

    @property
    def cx_partitions(self):
        return _DaskPartitionCoordinateIndexer(self, self.partition_sindex)

    def build_sindex(self, **kwargs):
        def build_sindex(series, **kwargs):
            series.build_sindex(**kwargs)
            return series
        return self.map_partitions(build_sindex, **kwargs, meta=self._meta)

    def intersects_bounds(self, bounds):
        return self.map_partitions(lambda s: s.intersects_bounds(bounds))

    # Override some standard Dask Series methods to propagate extra properties
    def _propagate_props_to_series(self, new_series):
        new_series._partition_bounds = self._partition_bounds
        new_series._partition_sindex = self._partition_sindex
        return new_series

    def persist(self, **kwargs):
        return self._propagate_props_to_series(
            super().persist(**kwargs)
        )


@make_meta_dispatch.register(GeoSeries)
def make_meta_series(s, index=None):
    result = s.head(0)
    if index is not None:
        result = result.reindex(index[:0])
    return result


@meta_nonempty.register(GeoSeries)
def meta_nonempty_series(s, index=None):
    return GeoSeries(make_array_nonempty(s.dtype), index=index)


@get_parallel_type.register(GeoSeries)
def get_parallel_type_dataframe(df):
    return DaskGeoSeries


class DaskGeoDataFrame(dd.DataFrame):
    def __init__(self, dsk, name, meta, divisions):
        super().__init__(dsk, name, meta, divisions)
        self._partition_sindex = {}
        self._partition_bounds = {}

    def to_parquet(self, fname, compression="snappy", filesystem=None, **kwargs):
        from .io import to_parquet_dask
        to_parquet_dask(
            self, fname, compression=compression, filesystem=filesystem, **kwargs
        )

    @property
    def geometry(self):
        # Use self._meta.geometry.name rather than self._meta._geometry so that an
        # informative error message is raised if there is no valid geometry column
        return self[self._meta.geometry.name]

    def set_geometry(self, geometry):
        if geometry != self._meta._geometry:
            return self.map_partitions(lambda df: df.set_geometry(geometry))
        else:
            return self

    @property
    def partition_sindex(self):
        geometry_name = self._meta.geometry.name
        if geometry_name not in self._partition_sindex:

            # Apply partition_bounds to geometry Series before creating the spatial
            # index. This removes the need to scan partitions to compute bounds
            geometry = self.geometry
            if geometry_name in self._partition_bounds:
                geometry._partition_bounds = self._partition_bounds[geometry_name]

            self._partition_sindex[geometry.name] = geometry.partition_sindex
            self._partition_bounds[geometry_name] = geometry.partition_bounds
        return self._partition_sindex[geometry_name]

    @property
    def cx(self):
        return _DaskCoordinateIndexer(self, self.partition_sindex)

    @property
    def cx_partitions(self):
        return _DaskPartitionCoordinateIndexer(self, self.partition_sindex)

    def pack_partitions(self, npartitions=None, p=15, shuffle='tasks'):
        """
        Repartition and reorder dataframe spatially along a Hilbert space filling curve

        Args:
            npartitions: Number of output partitions. Defaults to the larger of 8 and
                the length of the dataframe divided by 2**23.
            p: Hilbert curve p parameter
            shuffle: Dask shuffle method, either "disk" or "tasks"

        Returns:
            Spatially partitioned DaskGeoDataFrame
        """
        # Compute number of output partitions
        npartitions = self._compute_packing_npartitions(npartitions)

        # Add hilbert_distance column
        ddf = self._with_hilbert_distance_column(p)

        # Set index to distance. This will trigger an expensive shuffle
        # sort operation
        ddf = ddf.set_index('hilbert_distance', npartitions=npartitions, shuffle=shuffle)

        if ddf.npartitions != npartitions:
            # set_index doesn't change the number of partitions if the partitions
            # happen to be already sorted
            ddf = ddf.repartition(npartitions=npartitions)

        return ddf

    def pack_partitions_to_parquet(
        self,
        path,
        filesystem=None,
        npartitions=None,
        p=15,
        compression="snappy",
        tempdir_format=None,
        _retry_args=None,
        storage_options=None,
        engine_kwargs=None,
        overwrite=False,
    ):
        """
        Repartition and reorder dataframe spatially along a Hilbert space filling curve
        and write to parquet dataset at the provided path.

        This is equivalent to ddf.pack_partitions(...).to_parquet(...) but with lower
        memory and disk usage requirements.

        Args:
            path: Output parquet dataset path
            filesystem: Optional fsspec filesystem. If not provided, filesystem type
                is inferred from path
            npartitions: Number of output partitions. Defaults to the larger of 8 and
                the length of the dataframe divided by 2**23.
            p: Hilbert curve p parameter
            compression: Compression algorithm for parquet file
            tempdir_format: format string used to generate the filesystem path where
                temporary files should be stored for each output partition.  String
                must contain a '{partition}' replacement field which will be formatted
                using the output partition number as an integer. The string may
                optionally contain a '{uuid}' replacement field which will be formatted
                using a randomly generated UUID string. If None (the default),
                temporary files are stored inside the output path.

                These directories are deleted as soon as possible during the execution
                of the function.
            storage_options: Key/value pairs to be passed on to the file-system backend, if any.
            engine_kwargs: pyarrow.parquet engine-related keyword arguments.
        Returns:
            DaskGeoDataFrame backed by newly written parquet dataset
        """
        from .io import read_parquet, read_parquet_dask
        from .io.utils import validate_coerce_filesystem

        engine_kwargs = engine_kwargs or {}

        # Get fsspec filesystem object
        filesystem = validate_coerce_filesystem(path, filesystem, storage_options)

        # Decorator for operations that should be retried
        if _retry_args is None:
            _retry_args = dict(
                wait_exponential_multiplier=100,
                wait_exponential_max=120000,
                stop_max_attempt_number=24,
            )
        retryit = retry(**_retry_args)

        @retryit
        def rm_retry(file_path):
            filesystem.invalidate_cache()
            if filesystem.exists(file_path):
                filesystem.rm(file_path, recursive=True)
                if filesystem.exists(file_path):
                    # Make sure we keep retrying until file does not exist
                    raise ValueError("Deletion of {path} not yet complete".format(
                        path=file_path
                    ))

        @retryit
        def mkdirs_retry(dir_path):
            filesystem.makedirs(dir_path, exist_ok=True)

        # For filesystems that provide a "refresh" argument, set it to True
        if 'refresh' in signature(filesystem.ls).parameters:
            ls_kwargs = {'refresh': True}
        else:
            ls_kwargs = {}

        @retryit
        def ls_retry(dir_path):
            filesystem.invalidate_cache()
            return filesystem.ls(dir_path, **ls_kwargs)

        @retryit
        def move_retry(p1, p2):
            if filesystem.exists(p1):
                filesystem.move(p1, p2)

        # Compute tempdir_format string
        dataset_uuid = str(uuid.uuid4())
        if tempdir_format is None:
            tempdir_format = os.path.join(path, "part.{partition}.parquet")
        elif not isinstance(tempdir_format, str) or "{partition" not in tempdir_format:
            raise ValueError(
                "tempdir_format must be a string containing a {{partition}} "
                "replacement field\n"
                "    Received: {tempdir_format}".format(
                    tempdir_format=repr(tempdir_format)
                )
            )

        # Compute number of output partitions
        npartitions = self._compute_packing_npartitions(npartitions)
        out_partitions = list(range(npartitions))

        # Add hilbert_distance column
        ddf = self._with_hilbert_distance_column(p)

        # Compute output hilbert_distance divisions
        quantiles = partition_quantiles(
            ddf.hilbert_distance, npartitions
        ).compute().values

        # Add _partition column containing output partition number of each row
        ddf = ddf.map_partitions(
            lambda df: df.assign(
                _partition=np.digitize(df.hilbert_distance, quantiles[1:], right=True))
        )

        # Compute part paths
        parts_tmp_paths = [
            tempdir_format.format(partition=out_partition, uuid=dataset_uuid)
            for out_partition in out_partitions
        ]
        part_output_paths = [
            os.path.join(path, "part.%d.parquet" % out_partition)
            for out_partition in out_partitions
        ]

        # Initialize output partition directory structure
        filesystem.invalidate_cache()
        if overwrite:
            rm_retry(path)

        for out_partition in out_partitions:
            part_dir = os.path.join(path, "part.%d.parquet" % out_partition)
            mkdirs_retry(part_dir)
            tmp_part_dir = tempdir_format.format(partition=out_partition, uuid=dataset_uuid)
            mkdirs_retry(tmp_part_dir)

        # Shuffle and write a parquet dataset for each output partition
        @retryit
        def write_partition(df_part, part_path):
            with filesystem.open(part_path, "wb") as f:
                df_part.to_parquet(
                    f,
                    compression=compression,
                    index=True,
                    **engine_kwargs,
                )


        def process_partition(df, i):
            subpart_paths = {}
            for out_partition, df_part in df.groupby('_partition'):
                part_path = os.path.join(
                    tempdir_format.format(partition=out_partition, uuid=dataset_uuid),
                    'part.%d.parquet' % i,
                )
                df_part = (
                    df_part
                    .drop('_partition', axis=1)
                    .set_index('hilbert_distance', drop=True)
                )
                write_partition(df_part, part_path)
                subpart_paths[out_partition] = part_path

            return subpart_paths

        part_path_infos = dask.compute(*[
            dask.delayed(process_partition, pure=False)(df, i)
            for i, df in enumerate(ddf.to_delayed())
        ])

        # Build dict from part number to list of subpart paths
        part_num_to_subparts = {}
        for part_path_info in part_path_infos:
            for part_num, subpath in part_path_info.items():
                subpaths = part_num_to_subparts.get(part_num, [])
                subpaths.append(subpath)
                part_num_to_subparts[part_num] = subpaths

        # Concat parquet dataset per partition into parquet file per partition
        @retryit
        def write_concatted_part(part_df, part_output_path, md_list):
            with filesystem.open(part_output_path, 'wb') as f:
                pq.write_table(
                    pa.Table.from_pandas(part_df),
                    f, compression=compression, metadata_collector=md_list
                )

        @retryit
        def read_parquet_retry(parts_tmp_path, subpart_paths, part_output_path):
            if filesystem.isfile(part_output_path) and not filesystem.isdir(parts_tmp_path):
                # Handle rare case where the task was resubmitted and the work has
                # already been done.  This shouldn't happen with pure=False, but it
                # seems like it does very rarely.
                return read_parquet(
                    part_output_path,
                    filesystem=filesystem,
                    storage_options=storage_options,
                    **engine_kwargs,
                )

            ls_res = sorted(filesystem.ls(parts_tmp_path, **ls_kwargs))
            subpart_paths_stripped = sorted([filesystem._strip_protocol(_) for _ in subpart_paths])

            if subpart_paths_stripped != ls_res:
                missing = set(subpart_paths) - set(ls_res)
                extras = set(ls_res) - set(subpart_paths)
                raise ValueError(
                    "Filesystem not yet consistent\n"
                    "  Expected len: {expected}\n"
                    "  Found len: {received}\n"
                    "  Missing: {missing}\n"
                    "  Extras: {extras}".format(
                        expected=len(subpart_paths),
                        received=len(ls_res),
                        missing=list(missing),
                        extras=list(extras)
                    )
                )
            return read_parquet(
                parts_tmp_path,
                filesystem=filesystem,
                storage_options=storage_options,
                **engine_kwargs,
            )

        def concat_parts(parts_tmp_path, subpart_paths, part_output_path):
            filesystem.invalidate_cache()

            # Load directory of parquet parts for this partition into a
            # single GeoDataFrame
            if not subpart_paths:
                # Empty partition
                rm_retry(parts_tmp_path)
                return None
            else:
                part_df = read_parquet_retry(parts_tmp_path, subpart_paths, part_output_path)

            # Compute total_bounds for all geometry columns in part_df
            total_bounds = {}
            for series_name in part_df.columns:
                series = part_df[series_name]
                if isinstance(series.dtype, GeometryDtype):
                    total_bounds[series_name] = series.total_bounds

            # Delete directory of parquet parts for partition
            rm_retry(parts_tmp_path)
            rm_retry(part_output_path)

            # Sort by part_df by hilbert_distance index
            part_df.sort_index(inplace=True)

            # Write part_df as a single parquet file, collecting metadata for later use
            # constructing the full dataset _metadata file.
            md_list = []
            filesystem.invalidate_cache()
            write_concatted_part(part_df, part_output_path, md_list)

            # Return metadata and total_bounds for part
            return {"meta": md_list[0], "total_bounds": total_bounds}

        write_info = dask.compute(*[
            dask.delayed(concat_parts, pure=False)(
                parts_tmp_paths[out_partition],
                part_num_to_subparts.get(out_partition, []),
                part_output_paths[out_partition]
            )
            for out_partition in out_partitions
        ])

        # Handle empty partitions.
        input_paths, write_info = zip(*[
            (pp, wi) for (pp, wi) in zip(part_output_paths, write_info) if wi is not None
        ])
        output_paths = part_output_paths[:len(input_paths)]
        for p1, p2 in zip(input_paths, output_paths):
            if p1 != p2:
                move_retry(p1, p2)

        # Write _metadata
        meta = write_info[0]['meta']
        for i in range(1, len(write_info)):
            meta.append_row_groups(write_info[i]["meta"])

        @retryit
        def write_metadata_file():
            with filesystem.open(os.path.join(path, "_metadata"), 'wb') as f:
                meta.write_metadata_file(f)
        write_metadata_file()

        # Collect total_bounds per partition for all geometry columns
        all_bounds = {}
        for info in write_info:
            for col, bounds in info.get('total_bounds', {}).items():
                bounds_list = all_bounds.get(col, [])
                bounds_list.append(
                    pd.Series(bounds, index=['x0', 'y0', 'x1', 'y1'])
                )
                all_bounds[col] = bounds_list

        # Build spatial metadata for parquet dataset
        partition_bounds = {}
        for col, bounds in all_bounds.items():
            partition_bounds[col] = pd.DataFrame(all_bounds[col]).to_dict()

        spatial_metadata = {'partition_bounds': partition_bounds}
        b_spatial_metadata = json.dumps(spatial_metadata).encode('utf')

        # Write _common_metadata
        @retryit
        def write_commonmetadata_file():
            with filesystem.open(os.path.join(path, "part.0.parquet")) as f:
                pf = pq.ParquetFile(f)

            all_metadata = copy.copy(pf.metadata.metadata)
            all_metadata[b'spatialpandas'] = b_spatial_metadata

            new_schema = pf.schema.to_arrow_schema().with_metadata(all_metadata)
            with filesystem.open(os.path.join(path, "_common_metadata"), 'wb') as f:
                pq.write_metadata(new_schema, f)
        write_commonmetadata_file()

        return read_parquet_dask(
            path,
            filesystem=filesystem,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    def _compute_packing_npartitions(self, npartitions):
        if npartitions is None:
            # Make partitions of ~8 million rows with a minimum of 8
            # partitions
            nrows = len(self)
            npartitions = max(nrows // 2 ** 23, 8)
        return npartitions

    def _with_hilbert_distance_column(self, p):
        # Get geometry column
        geometry = self.geometry
        # Compute distance of points along the Hilbert-curve
        total_bounds = geometry.total_bounds
        ddf = self.assign(hilbert_distance=geometry.map_partitions(
            lambda s: s.hilbert_distance(total_bounds=total_bounds, p=p))
        )
        return ddf

    # Override some standard Dask Dataframe methods to propagate extra properties
    def _propagate_props_to_dataframe(self, new_frame):
        new_frame._partition_sindex = self._partition_sindex
        new_frame._partition_bounds = self._partition_bounds
        return new_frame

    def _propagate_props_to_series(self, new_series):
        if new_series.name in self._partition_bounds:
            new_series._partition_bounds = self._partition_bounds[new_series.name]
        if new_series.name in self._partition_sindex:
            new_series._partition_sindex = self._partition_sindex[new_series.name]
        return new_series

    def build_sindex(self, **kwargs):
        def build_sindex(df, **kwargs):
            df.build_sindex(**kwargs)
            return df
        return self.map_partitions(build_sindex, **kwargs, meta=self._meta)

    def persist(self, **kwargs):
        return self._propagate_props_to_dataframe(
            super().persist(**kwargs)
        )

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if np.isscalar(key) or isinstance(key, (tuple, str)):
            # New series of single column, partition props apply if we have them
            self._propagate_props_to_series(result)
        elif isinstance(key, (np.ndarray, list)):
            # New dataframe with same length and same partitions as self so partition
            # properties still apply
            self._propagate_props_to_dataframe(result)
        return result


@make_meta_dispatch.register(GeoDataFrame)
def make_meta_dataframe(df, index=None):
    result = df.head(0)
    if index is not None:
        result = result.reindex(index[:0])
    return result


@meta_nonempty.register(GeoDataFrame)
def meta_nonempty_dataframe(df, index=None):
    return GeoDataFrame(meta_nonempty(pd.DataFrame(df.head(0))))


@get_parallel_type.register(GeoDataFrame)
def get_parallel_type_series(s):
    return DaskGeoDataFrame


class _DaskCoordinateIndexer(_BaseCoordinateIndexer):
    def __init__(self, obj, sindex):
        super().__init__(sindex)
        self._obj = obj

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        covers_inds = set(covers_inds)
        overlaps_inds = set(overlaps_inds)
        all_partition_inds = sorted(covers_inds.union(overlaps_inds))
        if len(all_partition_inds) == 0:
            # No partitions intersect with query region, return empty result
            return dd.from_pandas(self._obj._meta, npartitions=1)

        @delayed
        def cx_fn(df):
            return df.cx[x0:x1, y0:y1]

        ddf = self._obj.partitions[all_partition_inds]
        delayed_dfs = []
        for partition_ind, delayed_df in zip(all_partition_inds, ddf.to_delayed()):
            if partition_ind in overlaps_inds:
                delayed_dfs.append(
                    cx_fn(delayed_df)
                )
            else:
                delayed_dfs.append(delayed_df)

        return dd.from_delayed(delayed_dfs, meta=ddf._meta, divisions=ddf.divisions)


class _DaskPartitionCoordinateIndexer(_BaseCoordinateIndexer):
    def __init__(self, obj, sindex):
        super().__init__(sindex)
        self._obj = obj

    def _perform_get_item(self, covers_inds, overlaps_inds, x0, x1, y0, y1):
        covers_inds = set(covers_inds)
        overlaps_inds = set(overlaps_inds)
        all_partition_inds = sorted(covers_inds.union(overlaps_inds))
        if len(all_partition_inds) == 0:
            # No partitions intersect with query region, return empty result
            return dd.from_pandas(self._obj._meta, npartitions=1)

        return self._obj.partitions[all_partition_inds]
