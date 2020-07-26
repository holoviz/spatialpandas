Version 0.3.6
=============

### Added
 - More intuitive error when lsuffix == rsuffix on sjoin ([#35](https://github.com/holoviz/spatialpandas/issues/35))

### Fixed
 - `read_parquet_dask` fails to read from s3 glob ([#34](https://github.com/holoviz/spatialpandas/issues/34))
 - Tests failing with `ValueError: Cannot mask with a boolean indexer containing NA values` ([#41](https://github.com/holoviz/spatialpandas/issues/41))
 - Tests in `test_parquet` failing with `TypeError: argument of type 'PosixPath' is not iterable` ([#42](https://github.com/holoviz/spatialpandas/issues/42))
 - Create temp directory for partitions explitictly, fixes failure of test `test_pack_partitions_to_parquet`

### Updated
 - Numba import updated to address deprecation warning ([#36](https://github.com/holoviz/spatialpandas/issues/36))


Version 0.3.5
=============

### Fixed
 - Fixed `GeoDataFrame` constructor exception when GeoPandas is not installed.

Version 0.3.4
=============

### Fixed
 - Support importing GeoPandas geometry series that contain `None` values.
 - Fixed `abstract://` protocol error in `pack_partitions_to_parquet` when run on
 local filesystem.
 - Preserve active geometry column when importing GeoPandas `GeoDataFrame`.
 - Always load index columns when the `columns` argument is passed to `read_parquet`.

### Updated
 - Added support for pandas 1.0.
 - Added support for pyarrow 0.16. When 0.16 is available, the performance of
 `read_parquet` and `read_parquet_dask` is significantly improved.


Version 0.3.2 / 0.3.3
=====================

### Fixed
 - Various reliability improvements for `pack_partitions_to_parquet`

Version 0.3.1
=============

### Fixed
 - Restored `categories` argument to `read_parquet_dask` function
 - Retry filesystem operations in `pack_partitions_to_parquet` using exponential backoff

Version 0.3.0
=============

### Added
 - Added partial support for the `intersects` geometry array method. Currently, it only
 supports being called on `Point` arrays, but the elements of the array can be compared to any scaler geometry object ([#21](https://github.com/holoviz/spatialpandas/pull/21)).
 - Added partial support for the `sjoin` spatial join function ([#21](https://github.com/holoviz/spatialpandas/pull/21)).
 - Added support for glob path strings, and lists of path strings, to the `read_parquet_dask` method ([#20](https://github.com/holoviz/spatialpandas/pull/20))
 - Added `bounds` argument to `read_parquet_dask` to support filtering the loaded partitions to those that intersect with a bounding box ([#20](https://github.com/holoviz/spatialpandas/pull/20))
 - Added `temp_format` argument to the `pack_partitions_to_parquet` method to control the location of temporary files ([#22](https://github.com/holoviz/spatialpandas/pull/22))


Version 0.2.0
=============

### Added
 - Added `pack_partitions_to_parquet` method to `DaskGeoDataFrame` ([#19](https://github.com/holoviz/spatialpandas/pull/19))
 - Added support for remote filesystems using the `fsspec` library ([#19](https://github.com/holoviz/spatialpandas/pull/19))

Version 0.1.1
=============

### Added
 - Documented dependencies required for the Overview notebook ([#18](https://github.com/holoviz/spatialpandas/pull/18))

### Fixed
 - Fixed Ring.to_shapely error ([#17](https://github.com/holoviz/spatialpandas/pull/17))

Version 0.1.0
=============

First public release available on PyPI and the pyviz anaconda channel.
