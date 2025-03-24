import numpy as np
import pytest

from pylib.auxlib._shift_no_wrap import shift_no_wrap


def test_shift_no_wrap_1d_positive():
    a = np.array([1, 2, 3, 4, 5])
    expected = np.array([0, 0, 1, 2, 3])
    result = shift_no_wrap(a, 2)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_1d_negative():
    a = np.array([1, 2, 3, 4, 5])
    expected = np.array([3, 4, 5, 0, 0])
    result = shift_no_wrap(a, -2)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_flatten():
    # When axis is None, the array is flattened.
    a = np.arange(10).reshape(2, 5)
    # Flattened a: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Shift by 3: first 3 become fill_value and then first 7 elements are copied.
    # Expected flattened result: [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    expected = np.array([[0, 0, 0, 0, 1], [2, 3, 4, 5, 6]])
    result = shift_no_wrap(a, 3)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_axis0_positive():
    a = np.arange(10).reshape(2, 5)
    # Shift rows (axis 0) down by 1: first row filled, second row equals original first row.
    expected = np.array([[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    result = shift_no_wrap(a, 1, axis=0)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_axis1_positive():
    a = np.arange(10).reshape(2, 5)
    # Shift columns (axis 1) right by 2: first two columns filled.
    expected = np.array([[0, 0, 0, 1, 2], [0, 0, 5, 6, 7]])
    result = shift_no_wrap(a, 2, axis=1)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_multi_axis():
    a = np.arange(10).reshape(2, 5)
    # Simultaneous shift on axes (0,1) with shifts (1,1)
    # For axis 0: shift down by 1, for axis 1: shift right by 1.
    # Expected: only the block at result[1, 1:5] is copied from a[0, 0:4].
    expected = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 3]])
    result = shift_no_wrap(a, (1, 1), axis=(0, 1))
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_zero_shift():
    # Zero shift should return the original array.
    a = np.random.randint(0, 100, (4, 4))
    result = shift_no_wrap(a, 0, axis=0)
    np.testing.assert_array_equal(result, a)


def test_shift_no_wrap_axis0_negative():
    a = np.arange(10).reshape(2, 5)
    # Shift rows (axis 0) up by 1 (negative shift): first row equals original second row.
    expected = np.array([[5, 6, 7, 8, 9], [0, 0, 0, 0, 0]])
    result = shift_no_wrap(a, -1, axis=0)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_axis1_negative():
    a = np.arange(10).reshape(2, 5)
    # Shift columns (axis 1) left by 2: first three columns are copied.
    expected = np.array([[2, 3, 4, 0, 0], [7, 8, 9, 0, 0]])
    result = shift_no_wrap(a, -2, axis=1)
    np.testing.assert_array_equal(result, expected)


def test_shift_no_wrap_invalid_shift_axis_length():
    a = np.arange(10).reshape(2, 5)
    # If shift and axis tuples are of different lengths, a ValueError should be raised.
    with pytest.raises(ValueError):
        shift_no_wrap(a, (1, 2), axis=(0,))
