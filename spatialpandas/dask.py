import dask.dataframe as dd

from spatialpandas.geometry.base import _BaseCoordinateIndexer
from spatialpandas.spatialindex import HilbertRtree
from .geoseries import GeoSeries
from .geodataframe import GeoDataFrame
from dask.dataframe.core import get_parallel_type
from dask.dataframe.utils import make_meta, meta_nonempty, make_array_nonempty
import pandas as pd
import numpy as np


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


@make_meta.register(GeoSeries)
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

    def to_parquet(self, fname, compression="snappy", **kwargs):
        from .io import to_parquet_dask
        to_parquet_dask(self, fname, compression=compression, **kwargs)

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

    def pack_partitions(self, p=10, npartitions=None, shuffle='tasks'):
        # Get geometry column
        geometry = self.geometry

        # Compute npartitions if needed
        if npartitions is None:
            # Make partitions of ~8 million rows with a minimum of 8
            # partitions
            nrows = len(self)
            npartitions = max(nrows // 2**23, 8)

        # Compute distance of points along the Hilbert-curve
        total_bounds = geometry.total_bounds
        ddf = self.assign(hilbert_distance=geometry.map_partitions(
            lambda s: s.hilbert_distance(total_bounds=total_bounds, p=p))
        )

        # Set index to distance. This will trigger an expensive shuffle
        # sort operation
        ddf = ddf.set_index('hilbert_distance', npartitions=npartitions, shuffle=shuffle)

        if ddf.npartitions != npartitions:
            # set_index doesn't change the number of partitions if the partitions
            # happen to be already sorted
            ddf = ddf.repartition(npartitions=npartitions)

        # Trigger calculation of partition bounds and spatial index
        ddf.partition_sindex

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


@make_meta.register(GeoDataFrame)
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

        result = self._obj.partitions[all_partition_inds]

        def map_fn(df, ind):
            if ind.iloc[0] in overlaps_inds:
                return df.cx[x0:x1, y0:y1]
            else:
                return df
        return result.map_partitions(
            map_fn,
            pd.Series(all_partition_inds, index=result.divisions[:-1]),
        )


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
