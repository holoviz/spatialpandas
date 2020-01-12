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
