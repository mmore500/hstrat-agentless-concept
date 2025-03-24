import numpy as np


def shift_no_wrap(a, shift, axis=None, fill_value=0):
    """
    Shift array elements along a given axis without wrap-around.

    Elements that would roll beyond the last position are replaced
    with the specified fill_value.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted. If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number. If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted. By default, the
        array is flattened before shifting, after which the original
        shape is restored.
    fill_value : scalar, optional
        The value used to fill the emptied entries. Default is 0.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> shift_no_wrap(a, 2)
    array([0, 0, 1, 2, 3])
    >>> shift_no_wrap(a, -2)
    array([3, 4, 5, 0, 0])

    >>> a2 = np.reshape(np.arange(10), (2, 5))
    >>> # Shift along the flattened array
    >>> shift_no_wrap(a2, 3)
    array([[0, 0, 0, 1, 2],
           [3, 4, 5, 6, 7]])
    >>> # Shift rows (axis=0) down by 1
    >>> shift_no_wrap(a2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> # Shift columns (axis=1) right by 2
    >>> shift_no_wrap(a2, 2, axis=1)
    array([[0, 0, 0, 1, 2],
           [0, 0, 5, 6, 7]])
    >>> # Simultaneous shift on multiple axes:
    >>> shift_no_wrap(a2, (1, 1), axis=(0, 1))
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 2, 3]])
    """
    a = np.asarray(a)

    # If no axis is provided, flatten the array, perform the shift, then reshape.
    if axis is None:
        original_shape = a.shape
        a_flat = a.flatten()
        n = a_flat.size
        shift_val = int(shift)
        result = np.empty_like(a_flat)
        if shift_val > 0:
            valid = max(n - shift_val, 0)
            result[:shift_val] = fill_value
            result[shift_val:] = a_flat[:valid]
        elif shift_val < 0:
            s_abs = -shift_val
            valid = max(n - s_abs, 0)
            result[:-s_abs] = a_flat[s_abs : s_abs + valid]
            result[-s_abs:] = fill_value
        else:
            result[:] = a_flat
        return result.reshape(original_shape)

    # Ensure axis and shift are tuples.
    if not isinstance(axis, tuple):
        axis = (axis,)
    if isinstance(shift, tuple):
        if len(shift) != len(axis):
            raise ValueError("shift and axis must be the same size")
    else:
        shift = (shift,) * len(axis)

    # Build a mapping from normalized axis to its shift value.
    ndim = a.ndim
    shift_dict = {}
    for ax, s in zip(axis, shift):
        ax = ax if ax >= 0 else ax + ndim
        shift_dict[ax] = s

    # Prepare slices for each axis.
    src_slices = []
    dest_slices = []
    for i, dim_size in enumerate(a.shape):
        if i in shift_dict:
            s = shift_dict[i]
            if s > 0:
                valid = max(dim_size - s, 0)
                src_slice = slice(0, valid)
                dest_slice = slice(s, s + valid)
            elif s < 0:
                s_abs = -s
                valid = max(dim_size - s_abs, 0)
                src_slice = slice(s_abs, s_abs + valid)
                dest_slice = slice(0, valid)
            else:
                src_slice = slice(0, dim_size)
                dest_slice = slice(0, dim_size)
        else:
            src_slice = slice(0, dim_size)
            dest_slice = slice(0, dim_size)
        src_slices.append(src_slice)
        dest_slices.append(dest_slice)

    # Create a result array filled with fill_value and copy the valid region.
    result = np.full(a.shape, fill_value, dtype=a.dtype)
    result[tuple(dest_slices)] = a[tuple(src_slices)]
    return result
