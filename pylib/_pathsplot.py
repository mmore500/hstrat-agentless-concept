import typing

import numpy as np
import pandas as pd
import seaborn as sns


def pathsplot(
    paths: typing.List[typing.List[typing.Tuple[int, int]]],
    xlim: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (
        None,
        None,
    ),
    ylim: typing.Tuple[typing.Optional[int], typing.Optional[int]] = (
        None,
        None,
    ),
    **kwargs: dict,
) -> sns.FacetGrid:
    """Plot multiple paths.

    Parameters
    ----------
    paths : List[List[Tuple[int, int]]]
        List of paths to plot.

    kwargs
        Additional keyword arguments to pass to seaborn.relplot.

    Returns
    -------
    sns.FacetGrid
        Seaborn FaceGrid object.
    """
    path_dfs = []
    for path in paths:
        x, y = zip(*path)
        x, y = [*x], [*y]
        t = np.arange(len(x) - 1)
        df = pd.concat(
            [
                pd.DataFrame({"x": x[:-1], "y": y[:-1], "t": t}),
                pd.DataFrame({"x": x[1:], "y": y[1:], "t": t}),
            ],
        )
        df["path"] = len(path_dfs)
        path_dfs.append(df)

    df = pd.concat(path_dfs)
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="t",
        col="path",
        estimator=None,
        kind="line",
        sort=False,
        **kwargs,
    )
    g.set(xlim=xlim, ylim=ylim)
    return g
