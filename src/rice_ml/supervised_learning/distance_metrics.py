"""
Vector distance utilities.

This module defines basic distance functions for numeric vectors,
implemented with NumPy and explicit input validation. The emphasis
is on predictable error messages and clean, readable logic.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = ["euclidean_distance", "manhattan_distance"]


# ---------------------------------------------------------------------
# Input normalization helpers
# ---------------------------------------------------------------------

def _as_1d_float(x, label: str) -> np.ndarray:
    """
    Convert input into a 1D NumPy array of floats.

    This helper enforces:
    - exactly one dimension
    - numeric values only
    - consistent error messages across distance functions

    Parameters
    ----------
    x : array_like
        Input vector.
    label : str
        Identifier used in error reporting.

    Returns
    -------
    ndarray
        1D array with dtype float.

    Raises
    ------
    ValueError
        If the input is not one-dimensional.
    TypeError
        If non-numeric data is encountered.
    """
    arr = np.asarray(x)

    # Shape validation first for consistent diagnostics
    if arr.ndim != 1:
        raise ValueError(
            f"Input '{label}' must be one-dimensional; received {arr.ndim}D."
        )

    # Explicit numeric check (handles object arrays cleanly)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"Input '{label}' must contain only numeric values."
        )

    try:
        return arr.astype(float, copy=False)
    except (TypeError, ValueError):
        raise TypeError(
            f"Input '{label}' must contain only numeric values."
        )


def _check_pair(a, b) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and align two vectors for distance computation.
    """
    a_vec = _as_1d_float(a, "a")
    b_vec = _as_1d_float(b, "b")

    if a_vec.shape != b_vec.shape:
        raise ValueError(
            f"Vector shapes must match: a.shape={a_vec.shape}, b.shape={b_vec.shape}."
        )

    return a_vec, b_vec


# ---------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------

def euclidean_distance(a, b) -> float:
    """
    Compute the L2 (Euclidean) distance between two vectors.

    This corresponds to the standard geometric distance in
    Euclidean space:

        √(∑ (aᵢ − bᵢ)²)

    Parameters
    ----------
    a : array_like
        First numeric vector.
    b : array_like
        Second numeric vector.

    Returns
    -------
    float
        Euclidean distance between `a` and `b`.

    Examples
    --------
    >>> euclidean_distance([0, 0], [3, 4])
    5.0
    >>> euclidean_distance([1, 2, 3], [1, 2, 3])
    0.0
    """
    a_vec, b_vec = _check_pair(a, b)
    return float(np.linalg.norm(a_vec - b_vec))


def manhattan_distance(a, b) -> float:
    """
    Compute the L1 (Manhattan) distance between two vectors.

    The Manhattan distance measures the total absolute deviation
    across all dimensions:

        ∑ |aᵢ − bᵢ|

    Parameters
    ----------
    a : array_like
        First numeric vector.
    b : array_like
        Second numeric vector.

    Returns
    -------
    float
        Manhattan distance between `a` and `b`.

    Examples
    --------
    >>> manhattan_distance([1, 2, 3], [4, 0, 3])
    5.0
    >>> manhattan_distance([0, 0], [0, 0])
    0.0
    """
    a_vec, b_vec = _check_pair(a, b)
    return float(np.sum(np.abs(a_vec - b_vec)))
