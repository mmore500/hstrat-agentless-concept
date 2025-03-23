import numpy as np


def transform_neighborhood_max(source_matrix, test_matrix, *, n):
    """
    For each position in source_matrix and test_matrix, finds the value in
    source_matrix corresponding to the location where test_matrix attains its
    maximum within the neighborhood of radius n.

    Parameters:
        source_matrix (np.ndarray): 2D array from which values are copied.
        test_matrix (np.ndarray): 2D array used to determine the maximum in each neighborhood.
                                   Must have the same shape as source_matrix.
        n (int): Number of steps in each direction (radius) for the neighborhood.

    Returns:
        np.ndarray: A 2D array of the same shape as source_matrix and
        test_matrix. For each position, it contains
                    the value from source_matrix at the position in the
                    neighborhood where test_matrix is maximal.
    """
    # Ensure source_matrix and test_matrix are 2D and of the same shape.
    if source_matrix.shape != test_matrix.shape or source_matrix.ndim != 2:
        raise ValueError(
            "source_matrix and test_matrix must be 2D arrays of the same shape."
        )

    m, p = test_matrix.shape
    window_size = 2 * n + 1

    # Pad test_matrix with -infinity so that out-of-bound areas never contribute to a maximum.
    padded_test = np.pad(
        test_matrix, pad_width=n, mode="constant", constant_values=-np.inf
    )

    # Create sliding windows view over padded_test.
    # The result has shape (m, p, window_size, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded_test, (window_size, window_size)
    )

    # Reshape the last two dimensions to flatten each window.
    flat_windows = windows.reshape(m, p, -1)

    # For each window, get the index of the maximum value.
    flat_idx = np.argmax(flat_windows, axis=-1)  # shape (m, p)

    # Convert flat indices to 2D coordinates within each window.
    offset_rows = flat_idx // window_size
    offset_cols = flat_idx % window_size

    # For each pixel (i, j) in the original image, its sliding window came from padded_test[i:i+window_size, j:j+window_size].
    # Thus, the global (padded) coordinates for the maximum are:
    global_rows = np.arange(m)[:, None] + offset_rows  # shape (m, p)
    global_cols = np.arange(p)[None, :] + offset_cols  # shape (m, p)

    # Adjust back to the coordinates of the original arrays by subtracting the pad width.
    orig_rows = global_rows - n
    orig_cols = global_cols - n

    # Now, use these coordinates to index into source_matrix.
    Z = source_matrix[orig_rows, orig_cols]
    return Z
