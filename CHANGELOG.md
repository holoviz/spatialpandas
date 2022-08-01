## Version 0.4.4

Date: 2022-08-01

This release achieves compatibility with recent versions of Dask, Pandas, PyArrow and Shapely.

Enhancements:

- Avoid inappropriate geometry tests ([#90](https://github.com/holoviz/spatialpandas/pull/90))

Compatibility:

- Fix Shapely deprecation warnings ([#85](https://github.com/holoviz/spatialpandas/pull/85))
- Pandas extension array fixes ([#88](https://github.com/holoviz/spatialpandas/pull/88))
- PyArrow and Dask parquet issues ([#92](https://github.com/holoviz/spatialpandas/pull/92))

## Version 0.4.3

Date: 2021-08-07

Enhancements:

- Expands the optional arguments that can be passed to `to_parquet`, `to_parquet_dask`, `read_parquet`, `read_parquet_dask` ensuring that `storage_options` is appropriately passed where needed.  ([#79](https://github.com/holoviz/spatialpandas/pull/79))
- Add typing information to some functions.
- Update `build_sindex` to pass through `kwargs` to underlying `HilbertRtree` implementation.
- Change `build_sindex` methods to consistently return reference to object to allow for method chaining.

Bug fixes:

- Update internal `validate_coerce_filesystem` to pass `storage_options` through. ([#78](https://github.com/holoviz/spatialpandas/pull/78))

Compatibility:

- Adds the ability to pass `storage_options` to the `to_parquet` function for `pandas > 1.2`, otherwise instantiates an `fsspec` filesystem with `storage_options` and passes that.
- Renames `fname` parameter to `path` to align with the pandas convention.


## Version 0.4.2

Date: 2021-07-28

This release primarily achieves compatibility with recent releases of Pandas. Many thanks to @Hoxbro for contributing the fixes and @philippjfr for ongoing maintenance of the project.

Compatibility:

- Compatibility with Pandas>=1.3 ([#76](https://github.com/holoviz/spatialpandas/pull/76))


## Version 0.4.1

Date: 2021-06-08

This release primarily achieves compatibility with recent releases of Dask. Many thanks to @jrbourbeau for contributing the fixes and @philippjfr for ongoing maintenance of the project.

Compatibility:

- Compatibility with Dask>=2021.06.0 ([#71](https://github.com/holoviz/spatialpandas/pull/71))


## Version 0.4.0

Date: 2021-05-25

Enhancements:

 - Add HoloViz build infrastructure. ([#50](https://github.com/holoviz/spatialpandas/pull/50))
 - Add `--skip-slow` and `--run-slow` options, slow tests are still run by default. ([#60](https://github.com/holoviz/spatialpandas/pull/60))
 - Add some type hints to parquet functions. ([#60](https://github.com/holoviz/spatialpandas/pull/60))
 - Allow using cx indexer without spatial index. ([#54](https://github.com/holoviz/spatialpandas/pull/54))
 - Switch to GitHub Actions. ([#55](https://github.com/holoviz/spatialpandas/pull/55))
 - Updated Geometry class __eq__ method so that if other object is a container, the equality method on the container is called, so now performing an equality check between a geometry object and a geometry array should return correct results, which should be a bool array, whereas previously it would simply return False because the objects were not the same immediate type. ([#60](https://github.com/holoviz/spatialpandas/pull/60))
 - Update GeometryArray class __eq__ method to allow comparison of an individual element to all objects in the array, returning an array of bool. ([#60](https://github.com/holoviz/spatialpandas/pull/60))
 - Add NotImplementedError for __contains__ method. ([#60](https://github.com/holoviz/spatialpandas/pull/60))

Bug fixes:

 - Fix compatibility with latest pandas. ([#55](https://github.com/holoviz/spatialpandas/pull/55))
 - Fix certain tests fail due to hypothesis health check. ([#60](https://github.com/holoviz/spatialpandas/pull/60))
 - Pin numpy and dask on MacOS to pass tests. ([#60](https://github.com/holoviz/spatialpandas/pull/60))


## Version 0.3.6

Date: 2020-08-16

Enhancements:

 - More intuitive error when lsuffix == rsuffix on sjoin ([#35](https://github.com/holoviz/spatialpandas/issues/35))

Bug fixes:

 - `read_parquet_dask` fails to read from s3 glob ([#34](https://github.com/holoviz/spatialpandas/issues/34))
 - Tests failing with `ValueError: Cannot mask with a boolean indexer containing NA values` ([#41](https://github.com/holoviz/spatialpandas/issues/41))
 - Tests in `test_parquet` failing with `TypeError: argument of type 'PosixPath' is not iterable` ([#42](https://github.com/holoviz/spatialpandas/issues/42))
 - Create temp directory for partitions explitictly, fixes failure of test `test_pack_partitions_to_parquet`

Compatibility:

 - Numba import updated to address deprecation warning ([#36](https://github.com/holoviz/spatialpandas/issues/36))


## Version 0.3.5

Date: 2020-07-27

Bug fixes:

 - Fixed `GeoDataFrame` constructor exception when GeoPandas is not installed.


## Version 0.3.4

Date: 2020-02-21

Bug fixes:

 - Support importing GeoPandas geometry series that contain `None` values.
 - Fixed `abstract://` protocol error in `pack_partitions_to_parquet` when run on
 local filesystem.
 - Preserve active geometry column when importing GeoPandas `GeoDataFrame`.
 - Always load index columns when the `columns` argument is passed to `read_parquet`.

Compatibility:

 - Added support for pandas 1.0.
 - Added support for pyarrow 0.16. When 0.16 is available, the performance of
 `read_parquet` and `read_parquet_dask` is significantly improved.


## Version 0.3.2 / 0.3.3

Date: 2020-01-24

Bug fixes:

 - Various reliability improvements for `pack_partitions_to_parquet`


## Version 0.3.1

Date: 2020-01-12

Bug fixes:

 - Restored `categories` argument to `read_parquet_dask` function
 - Retry filesystem operations in `pack_partitions_to_parquet` using exponential backoff


## Version 0.3.0

Date: 2020-01-11

Enhancements:

 - Added partial support for the `intersects` geometry array method. Currently, it only
 supports being called on `Point` arrays, but the elements of the array can be compared to any scaler geometry object ([#21](https://github.com/holoviz/spatialpandas/pull/21)).
 - Added partial support for the `sjoin` spatial join function ([#21](https://github.com/holoviz/spatialpandas/pull/21)).
 - Added support for glob path strings, and lists of path strings, to the `read_parquet_dask` method ([#20](https://github.com/holoviz/spatialpandas/pull/20))
 - Added `bounds` argument to `read_parquet_dask` to support filtering the loaded partitions to those that intersect with a bounding box ([#20](https://github.com/holoviz/spatialpandas/pull/20))
 - Added `temp_format` argument to the `pack_partitions_to_parquet` method to control the location of temporary files ([#22](https://github.com/holoviz/spatialpandas/pull/22))


## Version 0.2.0

Date: 2019-12-28

Enhancements:

 - Added `pack_partitions_to_parquet` method to `DaskGeoDataFrame` ([#19](https://github.com/holoviz/spatialpandas/pull/19))
 - Added support for remote filesystems using the `fsspec` library ([#19](https://github.com/holoviz/spatialpandas/pull/19))


## Version 0.1.1

Date: 2019-12-18

Enhancements:

 - Documented dependencies required for the Overview notebook ([#18](https://github.com/holoviz/spatialpandas/pull/18))

Bug fixes:

 - Fixed Ring.to_shapely error ([#17](https://github.com/holoviz/spatialpandas/pull/17))


## Version 0.1.0

Date: 2019-12-02

First public release available on PyPI and the pyviz anaconda channel.
