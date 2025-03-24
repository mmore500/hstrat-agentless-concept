import numpy as np

from pylib.auxlib._reverse_bits import reverse_bits


def test_reverse_bits_uint8():
    # For uint8: 1 (00000001) should become 128 (10000000)
    x = np.array([1, 2, 255], dtype=np.uint8)
    # 1 -> 128, 2 -> 64, 255 -> 255 (all ones remain the same)
    expected = np.array([128, 64, 255], dtype=np.uint8)
    result = reverse_bits(x)
    np.testing.assert_array_equal(result, expected)


def test_reverse_bits_uint16():
    # For uint16: 1 (00000000 00000001) should become 32768 (10000000 00000000)
    x = np.array([1, 2, 0xFF00], dtype=np.uint16)
    # 1 -> 32768, 2 -> 16384, and 0xFF00 (11111111 00000000) should reverse to 0x00FF (00000000 11111111)
    expected = np.array([32768, 16384, 0x00FF], dtype=np.uint16)
    result = reverse_bits(x)
    np.testing.assert_array_equal(result, expected)


def test_reverse_bits_uint32():
    # For uint32: 1 (in 32 bits) becomes 2147483648 (bit at MSB set)
    x = np.array([1, 2], dtype=np.uint32)
    expected = np.array([2147483648, 1073741824], dtype=np.uint32)
    result = reverse_bits(x)
    np.testing.assert_array_equal(result, expected)


def test_reverse_bits_uint64():
    # Specifically testing with np.uint64.
    # 1 in np.uint64 has a single 1 at the LSB; after reversing, it should be at the MSB,
    # which is 2**63 = 9223372036854775808.
    x = np.array([1, 2], dtype=np.uint64)
    expected = np.array(
        [9223372036854775808, 4611686018427387904], dtype=np.uint64
    )
    result = reverse_bits(x)
    np.testing.assert_array_equal(result, expected)


def test_reverse_bits_involution():
    # Check that reversing twice returns the original value.
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        # Create a test array with a variety of values.
        x = np.array(
            [0, 1, 2, 255, 1024, 2 ** (np.iinfo(dtype).bits - 1)], dtype=dtype
        )
        # Reversing the bits twice should give back the original numbers.
        np.testing.assert_array_equal(reverse_bits(reverse_bits(x)), x)
