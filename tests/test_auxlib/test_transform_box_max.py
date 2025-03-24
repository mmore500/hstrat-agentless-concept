import numpy as np
import pytest

from pylib.auxlib._transform_box_max import TransformBoxMax, transform_box_max


def test_transform_box_max_n0():
    # Setup simple 3x3 matrices.
    source_matrix = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
    )
    test_matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    expected = source_matrix.copy()

    result = transform_box_max(source_matrix, test_matrix, n=0)
    np.testing.assert_array_equal(result, expected)


def test_transform_box_max_n1():
    # Setup simple 3x3 matrices.
    source_matrix = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
    )
    test_matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    # Expected result calculated manually:
    # For position (0,0), the maximum in its neighborhood comes from source_matrix[1,1] = 50, etc.
    expected = np.array(
        [
            [50, 60, 60],
            [80, 90, 90],
            [80, 90, 90],
        ],
    )

    result = transform_box_max(source_matrix, test_matrix, n=1)
    np.testing.assert_array_equal(result, expected)


def test_transform_box_max_n2():
    # Create a 5x5 source matrix with distinct values.
    source_matrix = np.array(
        [
            [10, 20, 30, 40, 50],
            [60, 70, 80, 90, 100],
            [110, 120, 130, 140, 150],
            [160, 170, 180, 190, 200],
            [210, 220, 230, 240, 250],
        ]
    )

    # Create a corresponding 5x5 test matrix with increasing values.
    test_matrix = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=np.float64,
    )

    # Expected behavior:
    # For each (i,j), the maximum in the neighborhood (with radius 2) of test_matrix is at:
    # (min(4, i+2), min(4, j+2)) because the matrix increases monotonically.
    # Thus, the expected output is taken from source_matrix at those coordinates.
    expected = np.array(
        [
            [130, 140, 150, 150, 150],
            [180, 190, 200, 200, 200],
            [230, 240, 250, 250, 250],
            [230, 240, 250, 250, 250],
            [230, 240, 250, 250, 250],
        ]
    )

    result = transform_box_max(source_matrix, test_matrix, n=2)
    np.testing.assert_array_equal(result, expected)


def test_transform_box_max_invalid_shape():
    # Test for mismatched shapes.
    source_matrix = np.array([[1, 2, 3]])
    test_matrix = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        transform_box_max(source_matrix, test_matrix, n=1)


def test_transform_box_max_not_2d():
    # Test when inputs are not 2D.
    source_matrix = np.array([1, 2, 3])
    test_matrix = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        transform_box_max(source_matrix, test_matrix, n=2)


def test_TransformBoxMax_n1():
    # Setup simple 3x3 matrices.
    source_matrix = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ],
    )
    test_matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=np.float64,
    )

    # Expected result calculated manually:
    # For position (0,0), the maximum in its neighborhood comes from source_matrix[1,1] = 50, etc.
    expected = np.array(
        [
            [50, 60, 60],
            [80, 90, 90],
            [80, 90, 90],
        ],
    )

    result = TransformBoxMax(n=1)(source_matrix, test_matrix)
    np.testing.assert_array_equal(result, expected)
