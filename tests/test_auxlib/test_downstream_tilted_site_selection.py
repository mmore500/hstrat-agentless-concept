# adapted from https://github.com/mmore500/hstrat-surface-concept/blob/645a60e02fd05284929a3af763754acfb63f3614/implemented_pseudocode/test_tilted_site_selection.py
import functools
import itertools as it
import typing

from pylib.auxlib._downstream_tilted_site_selection import (
    _bit_floor,
    _ctz,
    _modpow2,
    downstream_tilted_site_selection,
)


def test_ctz():
    # fmt: off
    assert [*map(_ctz, range(1, 17))] == [
        0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4
    ]


def test_bit_floor():
    # fmt: off
    assert [*map(_bit_floor, range(1, 17))] == [
        1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16
    ]


def test_modpow2():
    assert _modpow2(10, 2) == 0  # 10 % 2 = 0
    assert _modpow2(10, 4) == 2  # 10 % 4 = 2
    assert _modpow2(10, 8) == 2  # 10 % 8 = 2
    assert _modpow2(15, 8) == 7  # 15 % 8 = 7
    assert _modpow2(20, 16) == 4  # 20 % 16 = 4
    assert _modpow2(16, 16) == 0  # 16 % 16 = 0
    assert _modpow2(1, 2) == 1  # 1 % 2 = 1
    assert _modpow2(3, 8) == 3  # 3 % 8 = 3
    assert _modpow2(1023, 1024) == 1023  # 1023 % 1024 = 1023
    assert _modpow2(0, 8) == 0  # 0 % 8 = 0


def validate_downstream_tilted_site_selection(
    fn: typing.Callable,
) -> typing.Callable:
    """Decorator to validate pre- and post-conditions on site selection."""

    @functools.wraps(fn)
    def wrapper(S: int, T: int) -> typing.Optional[int]:
        assert S.bit_count() == 1  # Assert S is a power of two
        assert S >= 8  # Assert S is at least 8
        assert 0 <= T  # Assert T is non-negative
        res = fn(S, T)
        assert 0 <= res < S  # Assert valid output
        return res

    return wrapper


site_selection = validate_downstream_tilted_site_selection(
    downstream_tilted_site_selection
)


def test_downstream_tilted_site_selection8():
    # fmt: off
    actual = (site_selection(8, T) for T in it.count())
    expected = [
        0, 1, 5, 2, 4, 6, 7, 3,  # T 0-7
        0, 1, 5, 7, 0, 6, 5, 4,  # T 8-15
        0, 1, 0, 2, 0, 6, 0, 3,  # T 16-23
        0, 1, 0, 7, 0, 6, 0, 5,  # T 24-31
        0, 1, 0, 2, 0, 1, 0, 3 # T 32-39
    ]
    assert all(x == y for x, y in zip(actual, expected))


def test_downstream_tilted_site_selection16():
    # fmt: off
    actual = (site_selection(16, T) for T in it.count())
    expected = [
        0, 1, 9, 2, 6, 10, 13, 3,  # T 0-7 --- hv 0,1,0,2,0,1,0,3
        5, 7, 8, 11, 12, 14, 15, 4,  # T 8-15 --- hv 0,1,0,2,0,1,0,4
        0, 1, 9, 8, 6, 10, 13, 12,  # T 16-24 --- hv 0,1,0, ...
        0, 7, 9, 15, 6, 14, 13, 5,  # T 24-31
        0, 1, 9, 2, 0, 10, 9, 3 # T 32-39
    ]
    assert all(x == y for x, y in zip(actual, expected))


def test_downstream_tilted_site_selection_fuzz():
    for S in (1 << s for s in range(3, 17)):
        for T in range(S - 1):
            site_selection(S, T)  # Validated via wrapper


def test_downstream_tilted_site_selection_epoch0():
    for S in (1 << s for s in range(3, 17)):
        actual = {site_selection(S, T) for T in range(S)}
        expected = set(range(S))
        assert actual == expected
