import numpy as np

from spatialpandas.geometry import PointArray


def test_construct_pointarray_interleaved():
    src_array = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
    points = PointArray(src_array)

    np.testing.assert_array_equal(points.x, src_array[0::2])
    np.testing.assert_array_equal(points.y, src_array[1::2])
    np.testing.assert_array_equal(points.isna(), np.isnan(src_array[0::2]))
    np.testing.assert_array_equal(points.flat_values, src_array)


def test_construct_pointarray_2d():
    src_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
    points = PointArray(src_array)

    np.testing.assert_array_equal(points.x, src_array[:, 0])
    np.testing.assert_array_equal(points.y, src_array[:, 1])
    np.testing.assert_array_equal(points.isna(), np.isnan(src_array[:, 0]))
    np.testing.assert_array_equal(points.flat_values, src_array.flatten())


def test_construct_pointarray_tuple():
    src_array = np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
    points = PointArray((src_array[:, 0], src_array[:, 1]))

    np.testing.assert_array_equal(points.x, src_array[:, 0])
    np.testing.assert_array_equal(points.y, src_array[:, 1])
    np.testing.assert_array_equal(points.isna(), np.isnan(src_array[:, 0]))
    np.testing.assert_array_equal(points.flat_values, src_array.flatten())


def test_construct_pointarray_2d_with_None():
    src_array = np.array([None, [2, 3], [4, 5], None], dtype=object)
    expected_xs = np.array([np.nan, 2, 4, np.nan], dtype=np.float64)
    expected_ys = np.array([np.nan, 3, 5, np.nan], dtype=np.float64)

    points = PointArray(src_array)

    np.testing.assert_array_equal(points.x, expected_xs)
    np.testing.assert_array_equal(points.y, expected_ys)
    np.testing.assert_array_equal(points.isna(), np.isnan(expected_xs))
    np.testing.assert_array_equal(
        points.flat_values[2:6], np.array([2, 3, 4, 5], dtype=np.int64)
    )
