import numpy as np


def randomize_nth_bit(
    arr,
    n,
    rand_val=None,
):
    """
    Randomizes the nth bit (0-indexed) of all integer values in a numpy array.

    For each element in the array, the nth bit is replaced with a random bit (0 or 1)
    while all other bits remain unchanged.

    Parameters:
        arr (np.ndarray): A numpy array of integers (any shape).
        n (int): The bit position to randomize (0-indexed, where 0 is the least significant bit).

    Returns:
        np.ndarray: A new numpy array with the nth bit of each element randomized.

    Raises:
        ValueError: If the input array is not of an integer type.
    """
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")

    # Generate random bits (0 or 1) for each element in the array.
    if rand_val is None:
        random_bits = np.random.randint(0, 2, size=arr.shape, dtype=arr.dtype)
    else:
        random_bits = np.full_like(arr, rand_val, dtype=arr.dtype)

    # Clear the nth bit in each element.
    cleared = arr & ~(np.ones_like(arr) << n)

    # Set the nth bit to the random bit.
    randomized = cleared | (random_bits << n)

    return randomized
