import typing

import pandas as pd
import seaborn as sns

from ._dewrap_path import dewrap_path


def pathsplot(
    paths: typing.List[typing.List[typing.Tuple[int, int]]],
    xmax: typing.Optional[int] = None,
    ymax: typing.Optional[int] = None,
    *,
    aspect: float = 1.0,
    height: float = 1.5,
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
    for p, path in enumerate(paths):
        for path in dewrap_path(path, xmax, ymax):
            t, x, y = zip(*path)
            t, x, y = [*t], [*x], [*y]
            df = pd.concat(
                [
                    pd.DataFrame({"x": x[:-1], "y": y[:-1], "t": t[:-1]}),
                    pd.DataFrame({"x": x[1:], "y": y[1:], "t": t[:-1]}),
                ],
            )
            df["path"] = p
            path_dfs.append(df)

    df = pd.concat(path_dfs)

    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="t",
        col="path",
        aspect=aspect,
        estimator=None,
        kind="line",
        height=height,
        sort=False,
        **kwargs,
    )
    g.set(xlim=(0, xmax), ylim=(ymax, 0))  # inverted y axes
    g.tight_layout()
    return g
