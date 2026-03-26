"""
Plotting utilities shared across rhoPCA classes.
"""

import warnings

import numpy as np
import seaborn as sns


_DEFAULT_CONTINUOUS_PALETTE = 'viridis'


def check_discrete(values, column):
    """
    Raise ``ValueError`` if *values* appear to be continuous.

    Only discrete (categorical, object, or integer) columns are supported for
    scatter and histogram plots.

    Parameters
    ----------
    values : array-like
        Column values to check.
    column : str
        Column name, used in the error message.
    """
    dtype = np.asarray(values).dtype
    if np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"color_by='{column}' contains floating-point values. "
            "Only discrete (categorical or integer) columns are supported for "
            "scatter and histogram plots."
        )


def resolve_palette(palette, n):
    """
    Return a list of *n* discrete colors from *palette*.

    If *palette* is a list shorter than *n*, the remaining slots are filled
    from the ``Set2`` palette.

    Parameters
    ----------
    palette : str or list
        Seaborn palette name or list of color specs.
    n : int
        Number of colors needed.

    Returns
    -------
    colors : list
    """
    if isinstance(palette, list):
        colors = list(palette)
        if len(colors) < n:
            warnings.warn(
                f"palette has {len(colors)} color(s) but {n} are needed. "
                f"Padding with {n - len(colors)} color(s) from Set2."
            )
            fallback = sns.color_palette("Set2", n_colors=n - len(colors))
            colors = colors + list(fallback)
        return colors[:n]
    return list(sns.color_palette(palette, n_colors=n))


def resolve_continuous_palette(palette):
    """
    Return a colormap name suitable for continuous data.

    If *palette* is a list (intended for discrete use), it is ignored and the
    default continuous palette (``'viridis'``) is returned instead.

    Parameters
    ----------
    palette : str or list
        User-supplied palette argument.

    Returns
    -------
    cmap : str
        Matplotlib colormap name.
    """
    if isinstance(palette, list):
        return _DEFAULT_CONTINUOUS_PALETTE
    return palette
