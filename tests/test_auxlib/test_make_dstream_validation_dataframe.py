import pandas as pd

from pylib.auxlib._hstrat_reconstruct_dataframe import (
    hstrat_reconstruct_dataframe,
)
from pylib.auxlib._make_dstream_validation_dataframe import (
    make_dstream_validation_dataframe,
)


def test_make_dstream_validation_dataframe():
    S = 64
    dstream_algo = "dstream.tilted_algo"

    # block validation setup
    block_validator = (
        f"pl.col('dstream_value') == pl.col('dstream_Tbar') // {S // 2 + 1}"
    )

    def block_differentia_override(T):
        return T // (S // 2 + 1)

    block_df1 = make_dstream_validation_dataframe(
        differentia_override=block_differentia_override,
        validator=block_validator,
        S=S,
        T=64,
        dstream_algo=dstream_algo,
    )
    block_df2 = make_dstream_validation_dataframe(
        differentia_override=block_differentia_override,
        validator=block_validator,
        S=S,
        T=65,
        dstream_algo=dstream_algo,
    )

    # checkerboard validation setup
    checkerboard_validator = (
        "pl.col('dstream_value') == pl.col('dstream_Tbar') % 2"
    )

    def checkerboard_differentia_override(T):
        return T % 2

    checkerboard_df1 = make_dstream_validation_dataframe(
        differentia_override=checkerboard_differentia_override,
        validator=checkerboard_validator,
        S=S,
        T=64,
        dstream_algo=dstream_algo,
    )
    checkerboard_df2 = make_dstream_validation_dataframe(
        differentia_override=checkerboard_differentia_override,
        validator=checkerboard_validator,
        S=S,
        T=65,
        dstream_algo=dstream_algo,
    )

    df = pd.concat([block_df1, block_df2, checkerboard_df1, checkerboard_df2])

    res = hstrat_reconstruct_dataframe(df)
    assert len(res) >= len(df)
    assert "ancestor_id" in res.columns
    assert "id" in res.columns


def test_make_dstream_validation_dataframe_trivial():
    S = 64
    dstream_algo = "dstream.tilted_algo"

    # trivial validator 0 setup
    trivial0_validator = "pl.col('dstream_value') == 0"

    def trivial_differentia_override0(T):
        return 0

    trivial0_df1 = make_dstream_validation_dataframe(
        differentia_override=trivial_differentia_override0,
        validator=trivial0_validator,
        S=S,
        T=64,
        dstream_algo=dstream_algo,
    )
    trivial0_df2 = make_dstream_validation_dataframe(
        differentia_override=trivial_differentia_override0,
        validator=trivial0_validator,
        S=S,
        T=65,
        dstream_algo=dstream_algo,
    )

    # trivial validator 1 setup
    trivial1_validator = "pl.col('dstream_value') == 1"

    def trivial_differentia_override1(T):
        return 1

    trivial1_df1 = make_dstream_validation_dataframe(
        differentia_override=trivial_differentia_override1,
        validator=trivial1_validator,
        S=S,
        T=64,
        dstream_algo=dstream_algo,
    )
    trivial1_df2 = make_dstream_validation_dataframe(
        differentia_override=trivial_differentia_override1,
        validator=trivial1_validator,
        S=S,
        T=65,
        dstream_algo=dstream_algo,
    )

    df = pd.concat([trivial0_df1, trivial0_df2, trivial1_df1, trivial1_df2])

    res = hstrat_reconstruct_dataframe(df)
    assert len(res) >= len(df)
    assert "ancestor_id" in res.columns
    assert "id" in res.columns
