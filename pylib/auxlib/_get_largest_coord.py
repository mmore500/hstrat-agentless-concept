import numpy as np


def get_largest_coord(arr, coord, n):
    """
    Get the (x, y) coordinate of the largest value within n steps of coord.

    Parameters:
        arr (np.ndarray): A 2D NumPy array.
        coord (tuple): The (x, y) coordinate (row, column) to center the search.
        n (int): Number of steps to search in each direction.

    Returns:
        tuple: Global (x, y) coordinate of the maximum value within the search window.
    """
    if np.max(arr.ravel()) == np.min(arr.ravel()):
        return coord
    x, y = coord

    # Determine the boundaries of the subarray, ensuring we stay within array limits.
    x_start = max(0, x - n)
    x_end = min(arr.shape[0], x + n + 1)
    y_start = max(0, y - n)
    y_end = min(arr.shape[1], y + n + 1)

    # Extract the subarray.
    subarr = arr[x_start:x_end, y_start:y_end]

    # Find the index of the maximum value within the subarray.
    flat_index = np.argmax(subarr)

    # Convert the flat index to 2D coordinates within the subarray.
    local_max_coord = np.unravel_index(flat_index, subarr.shape)

    # Adjust local coordinates to get the global coordinates.
    global_max_coord = (
        local_max_coord[0] + x_start,
        local_max_coord[1] + y_start,
    )

    if arr[global_max_coord] == arr[coord]:
        return coord

    return global_max_coord
