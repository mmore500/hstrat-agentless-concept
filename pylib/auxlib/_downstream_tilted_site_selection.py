# adapted from https://github.com/mmore500/hstrat-surface-concept/blob/645a60e02fd05284929a3af763754acfb63f3614/implemented_pseudocode/tilted_site_selection.py
def _modpow2(dividend: int, divisor: int) -> int:
    """Perform fast mod using bitwise operations.

    Parameters
    ----------
    dividend : int
        The dividend of the mod operation. Must be a positive integer.
    divisor : int
        The divisor of the mod operation. Must be a positive integer and a
        power of 2.

    Returns
    -------
    int
        The remainder of dividing the dividend by the divisor.
    """
    return dividend & (divisor - 1)


def _ctz(x: int) -> int:
    """Count trailing zeros."""
    assert x > 0
    return (x & -x).bit_length() - 1


def _bit_floor(n: int) -> int:
    """Calculate the largest power of two not greater than n.

    If zero, returns zero.
    """
    mask = 1 << (n >> 1).bit_length()
    return n & mask


def downstream_tilted_site_selection(S: int, T: int) -> int:
    """Site selection algorithm for tilted curation.

    Parameters
    ----------
    S : int
        Buffer size. Must be a power of two.
    T : int
        Current logical time. Must be less than 2**S - 1.

    Returns
    -------
    typing.Optional[int]
        Selected site, if any.
    """
    s = S.bit_length() - 1
    t = max((T).bit_length() - s, 0)  # Current epoch
    h = _ctz(T + 1)  # Current hanoi value
    i = T >> (h + 1)  # Hanoi value incidence (i.e., num seen)

    blt = t.bit_length()  # Bit length of t
    epsilon_tau = _bit_floor(t << 1) > t + blt  # Correction factor
    tau = blt - epsilon_tau  # Current meta-epoch
    t_0 = (1 << tau) - tau  # Opening epoch of meta-epoch
    t_1 = (1 << (tau + 1)) - (tau + 1)  # Opening epoch of next meta-epoch
    epsilon_b = t < h + t_0 < t_1  # Uninvaded correction factor
    B = S >> (tau + 1 - epsilon_b) or 1  # Num bunches available to h.v.

    b_l = _modpow2(i, B)  # Logical bunch index...
    # ... i.e., in order filled (increasing nestedness/decreasing init size r)

    # Need to calculate physical bunch index...
    # ... i.e., position among bunches left-to-right in buffer space
    v = b_l.bit_length()  # Nestedness depth level of physical bunch
    w = (S >> v) * bool(v)  # Num bunches spaced between bunches in nest level
    o = w >> 1  # Offset of nestedness level in physical bunch order
    p = b_l - _bit_floor(b_l)  # Bunch position within nestedness level
    b_p = o + w * p  # Physical bunch index...
    # ... i.e., in left-to-right sequential bunch order

    # Need to calculate buffer position of b_p'th bunch
    epsilon_k_b = bool(b_l)  # Correction factor for zeroth bunch...
    # ... i.e., bunch r=s at site k=0
    k_b = (  # Site index of bunch
        (b_p << 1) + ((S << 1) - b_p).bit_count() - 1 - epsilon_k_b
    )

    return k_b + h  # Calculate placement site...
    # ... where h.v. h is offset within bunch
