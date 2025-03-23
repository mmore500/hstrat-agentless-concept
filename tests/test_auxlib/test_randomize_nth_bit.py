import numpy as np
import pytest

from pylib.auxlib._randomize_nth_bit import randomize_nth_bit


def test_randomize_nth_bit_only_changes_nth_bit():
    # Use a fixed seed for reproducibility.
    np.random.seed(0)
    original = np.array([[5, 7, 8], [3, 12, 10]], dtype=np.int32)
    n = 1  # bit index to randomize (0-indexed)
    result = original.copy()
    while (result == original).ravel().all():
        result = randomize_nth_bit(original, n)

    # Create mask for the nth bit.
    mask = 1 << n

    # Check each element to ensure only the nth bit may have changed.
    for orig, new in zip(original.flatten(), result.flatten()):
        # The parts of the number other than the nth bit should remain the same.
        unchanged_orig = orig & ~mask
        unchanged_new = new & ~mask
        assert unchanged_orig == unchanged_new


def test_randomize_nth_bit_with_non_integer_input():
    # Test that a ValueError is raised when input array is not of integer type.
    non_int_array = np.array([1.1, 2.2, 3.3])
    with pytest.raises(ValueError):
        randomize_nth_bit(non_int_array, 1)
