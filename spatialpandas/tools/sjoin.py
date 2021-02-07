import numpy as np
import pandas as pd
from dask import delayed
from dask.dataframe import from_delayed, from_pandas


def _record_reset_index(df, suffix):
    # Record original index name(s), generate new index name(s), reset index column(s)
    new_column_name = ["index_%s" % suffix]
    df = df.copy(deep=True)
    try:
        old_index_name = [df.index.name]
        df.index = df.index.rename(new_column_name[0])
    except TypeError:
        new_column_name = [
            "index_%s" % suffix + str(l) for l, ix in
            enumerate(df.index.names)
        ]
        old_index_name = df.index.names
        df.index = df.index.rename(new_column_name)
    df = df.reset_index(drop=False)

    return df, old_index_name, new_column_name


def sjoin(
        left_df, right_df, how="inner", op="intersects",
        lsuffix="left", rsuffix="right"
):
    """
    Spatial join between two GeoDataFrames or between a DaskGeoDataFrame and
    a GeoDataFrame

    Args:
        left_df: A GeoDataFrame or DaskGeoDataFrame
        right_df: A GeoDataFrame
        how: The type of join. One of 'inner', 'left', or 'right'. Note that 'right'
            is not supported when left_df is a DaskGeoDataFrame
        op: Binary predicate, currently only "intersects" is supported
        lsuffix: Suffix to apply to overlapping column names from the left GeoDataFrame
        rsuffix: Suffix to apply to overlapping column names from the right GeoDataFrame

    Returns:
        GeoDataFrame or DaskGeoDataFrame (same type as left_df argument)
    """
    from .. import GeoDataFrame
    try:
        from ..dask import DaskGeoDataFrame
    except ImportError:
        DaskGeoDataFrame = type(None)

    # Validate data frame types
    if not isinstance(left_df, (GeoDataFrame, DaskGeoDataFrame)):
        raise ValueError(
            "`left_df` must be a spatialpandas.GeoDataFrame or "
            "spatialpandas.dask.DaskGeoDataFrame\n"
            "    Received value of type: {typ}".format(typ=type(left_df)))

    if not isinstance(right_df, GeoDataFrame):
        raise ValueError(
            "`right_df` must be a spatialpandas.GeoDataFrame\n"
            "    Received value of type: {typ}".format(typ=type(right_df)))

    # Validate op
    valid_op = ["intersects"]
    if op not in valid_op:
        raise ValueError(
            "`op` must be one of {valid_op}\n"
            "    Received: {val}".format(val=repr(how), valid_op=valid_op)
        )

    # Validate join type
    valid_how = ["left", "right", "inner"]
    if how not in valid_how:
        raise ValueError(
            "`how` must be one of {valid_how}\n"
            "    Received: {val}".format(val=repr(how), valid_how=valid_how)
        )

    # Validate suffixes
    if lsuffix == rsuffix:
        raise ValueError(
            "`lsuffix` and `rsuffix` must not be equal"
        )

    # Perform sjoin
    if isinstance(left_df, GeoDataFrame):
        return _sjoin_pandas_pandas(
            left_df, right_df, how=how, op=op, lsuffix=lsuffix, rsuffix=rsuffix
        )
    elif isinstance(left_df, DaskGeoDataFrame):
        return _sjoin_dask_pandas(
            left_df, right_df, how=how, op=op, lsuffix=lsuffix, rsuffix=rsuffix
        )


def _sjoin_dask_pandas(
        left_ddf, right_df, how="inner", op="intersects",
        lsuffix="left", rsuffix="right"
):
    if how == "right":
        raise ValueError(
            "`how` may not be 'right' when left_df is a DaskGeoDataFrame"
        )
    dfs = left_ddf.to_delayed()
    partition_bounds = left_ddf.geometry.partition_bounds
    sjoin_pandas = delayed(_sjoin_pandas_pandas)

    # Get spatial index for right_df
    right_sindex = right_df.geometry.sindex

    # Build list of delayed sjoin results
    joined_dfs = []
    for df, (i, bounds) in zip(dfs, partition_bounds.iterrows()):
        right_inds = right_sindex.intersects(bounds.values)
        if how == "left" or len(right_inds) > 0:
            joined_dfs.append(
                sjoin_pandas(
                    df, right_df.iloc[right_inds], how=how,
                    rsuffix=rsuffix, lsuffix=lsuffix
                )
            )

    # Compute meta
    meta = _sjoin_pandas_pandas(
        left_ddf._meta, right_df.iloc[:0], how=how, lsuffix=lsuffix, rsuffix=rsuffix
    )

    # Build resulting Dask DataFrame
    if not joined_dfs:
        return from_pandas(meta, npartitions=1)
    else:
        return from_delayed(joined_dfs, meta=meta)


def _sjoin_pandas_pandas(
        left_df, right_df, how="inner", op="intersects",
        lsuffix="left", rsuffix="right"
):
    from .. import GeoDataFrame

    # Record original index name(s), generate new index name(s), reset index column(s)
    original_right_df = right_df
    original_left_df = left_df
    right_df, right_index_name, index_right = _record_reset_index(
        original_right_df, rsuffix
    )
    left_df, left_index_name, index_left = _record_reset_index(
        original_left_df, lsuffix
    )

    if any(original_left_df.columns.isin(index_left + index_right)) or any(
            original_right_df.columns.isin(index_left + index_right)
    ):
        raise ValueError(
            "'{0}' and '{1}' cannot be column names in the GeoDataFrames being"
            " joined".format(index_left, index_right)
        )

    # Get spatial index for left frame
    sindex = left_df.geometry.sindex
    left_geom = left_df.geometry.array
    right_geom = right_df.geometry.array

    # Get bounds from right geometry
    right_bounds = right_df.geometry.bounds.values

    # Init list of arrays, the same length as right_df, where each array holds the
    # indices into left_df that intersect with the corresponding element.
    left_inds = [np.array([], dtype='uint32')] * len(right_df)

    # right_inds will hold the inds into right_df that correspond to left_inds
    right_inds = [np.array([], dtype='uint32')] * len(right_df)

    # Loop over the right frame
    for i in range(len(right_df)):
        # Get bounds for shape in current row of right_df
        shape_bounds = right_bounds[i, :]

        # Use spatial index on left_df to get indices of shapes with bounding boxes that
        # intersect with these bounds
        candidate_inds = sindex.intersects(shape_bounds)

        if len(candidate_inds) > 0:
            right_shape = right_geom[i]
            intersecting_mask = left_geom.intersects(right_shape, inds=candidate_inds)
            intersecting_inds = candidate_inds[intersecting_mask]
            left_inds[i] = intersecting_inds
            right_inds[i] = np.full(len(intersecting_inds), i)

    # Flatten nested arrays of indices
    if left_inds:
        flat_left_inds = np.concatenate(left_inds)
        flat_right_inds = np.concatenate(right_inds)
    else:
        flat_left_inds = np.array([], dtype='uint32')
        flat_right_inds = np.array([], dtype='uint32')

    # Build pandas DataFrame from inds
    result = pd.DataFrame({
        '_key_left': flat_left_inds,
        '_key_right': flat_right_inds
    })

    # Perform join
    if how == "inner":
        result = result.set_index("_key_left")
        joined = (
            left_df.merge(
                result, left_index=True, right_index=True
            ).merge(
                right_df.drop(right_df.geometry.name, axis=1),
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
            ).set_index(
                index_left
            ).drop(
                ["_key_right"], axis=1
            )
        )
        if len(left_index_name) > 1:
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name[0]

    elif how == "left":
        result = result.set_index("_key_left")
        joined = (
            left_df.merge(
                result, left_index=True, right_index=True, how="left"
            ).merge(
                right_df.drop(right_df.geometry.name, axis=1),
                how="left",
                left_on="_key_right",
                right_index=True,
                suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
            ).set_index(
                index_left
            ).drop(
                ["_key_right"], axis=1
            )
        )
        if len(left_index_name) > 1:
            joined.index.names = left_index_name
        else:
            joined.index.name = left_index_name[0]

    else:  # how == 'right':
        joined = (
            left_df.drop(
                left_df.geometry.name, axis=1
            ).merge(
                result.merge(
                    right_df, left_on="_key_right", right_index=True, how="right"
                ),
                left_index=True,
                right_on="_key_left",
                suffixes=("_%s" % lsuffix, "_%s" % rsuffix),
                how="right",
            ).set_index(
                index_right
            ).drop(
                ["_key_left", "_key_right"], axis=1
            )
        )
        if len(right_index_name) > 1:
            joined.index.names = right_index_name
        else:
            joined.index.name = right_index_name[0]

    return GeoDataFrame(joined)
