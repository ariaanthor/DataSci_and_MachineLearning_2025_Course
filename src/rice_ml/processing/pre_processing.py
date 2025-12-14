"""
pre_processing.py

This module contains a small set of “batteries-included” preprocessing tools
that are designed to be easy to test and easy to reason about:

- Feature-wise scaling/normalization (standardization, min-max, max-abs)
- Row-wise L2 normalization (useful for vector embeddings)
- Reproducible dataset splits (train/test and train/val/test), with optional stratification

The functions avoid external ML dependencies and emphasize:
- consistent input validation
- predictable numeric casting rules
- safe handling of zero-variance / zero-range columns

Functions
---------
standardize
    Feature-wise z-score transformation.
minmax_scale
    Feature-wise affine scaling into a given range.
maxabs_scale
    Feature-wise division by maximum absolute value.
l2_normalize_rows
    Row-wise L2 normalization for matrices.
train_test_split
    Split X (and optional y) into train/test.
train_val_test_split
    Split X (and optional y) into train/val/test.

Examples
--------
Standardization with a constant column:

>>> import numpy as np
>>> X = np.array([[1., 10.],
...               [2., 10.],
...               [3., 10.]], dtype=float)
>>> Z, params = standardize(X, return_params=True)
>>> Z[:, 1].tolist()  # constant column becomes all zeros
[0.0, 0.0, 0.0]
>>> params["scale"][1]  # scale=1 avoids division-by-zero
1.0

Row-wise L2 normalization:

>>> X = np.array([[3., 4.],
...               [0., 0.]], dtype=float)
>>> Xn = l2_normalize_rows(X)
>>> round(float(np.linalg.norm(Xn[0])), 6)
1.0
>>> Xn[1].tolist()
[0.0, 0.0]

Stratified train/test split:

>>> X = np.arange(20, dtype=float).reshape(10, 2)
>>> y = np.array([0]*5 + [1]*5)
>>> Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
>>> (yte.tolist().count(0), yte.tolist().count(1))
(1, 1)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

__all__ = [
    'ArrayLike',
    'standardize',
    'minmax_scale',
    'maxabs_scale',
    'l2_normalize_rows',
    'train_test_split',
    'train_val_test_split',
]

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# ---------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------
def _ensure_2d_numeric(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Coerce an input into a non-empty 2D float array.

    Notes
    -----
    - 2D shape is enforced because these utilities operate on matrices.
    - Non-numeric dtypes are rejected with a TypeError (after a best-effort cast).
    """
    arr = np.asarray(X)

    # Shape checks first so users get consistent, actionable errors.
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    # Numeric enforcement: either already numeric or castable to float.
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)

    return arr


def _ensure_1d_vector(y: Optional[ArrayLike], name: str = "y") -> Optional[np.ndarray]:
    """
    Ensure y is a 1D vector (or None).

    Notes
    -----
    - For splitting/stratification, labels can be any dtype (e.g., strings).
    - For numeric computations (scaling), callers should explicitly cast.
    """
    if y is None:
        return None
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _check_Xy_shapes(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Verify X and y agree on the number of samples (rows)."""
    if y is not None and len(y) != X.shape[0]:
        raise ValueError(
            f"X and y must have compatible first dimension; got len(y)={len(y)} "
            f"and X.shape[0]={X.shape[0]}."
        )


def _rng_from_seed(random_state: Optional[int]) -> np.random.Generator:
    """Create a NumPy Generator from an integer seed (or default RNG if None)."""
    if random_state is None:
        return np.random.default_rng()
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")
    return np.random.default_rng(int(random_state))


# ---------------------------------------------------------------------
# Scaling / Normalization
# ---------------------------------------------------------------------
def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Feature-wise z-score standardization.

    When enabled, each feature column is transformed as:
        (x - mean) / std

    If a column has zero standard deviation and `with_std=True`, the column is
    left at 0 (by using a scale factor of 1.0 for that feature).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    with_mean : bool, default=True
        Subtract the per-feature mean.
    with_std : bool, default=True
        Divide by the per-feature standard deviation.
    ddof : int, default=0
        Delta degrees of freedom for std calculation.
    return_params : bool, default=False
        If True, also return a dict containing 'mean' and 'scale'.

    Returns
    -------
    X_out : ndarray, shape (n_samples, n_features)
        Standardized array.
    params : dict, optional
        Only when return_params=True.

    Raises
    ------
    ValueError
        If X is not 2D or is empty.
    TypeError
        If X cannot be coerced to numeric floats.
    """
    X = _ensure_2d_numeric(X, "X")

    mean = X.mean(axis=0) if with_mean else np.zeros(X.shape[1], dtype=float)
    Xc = X - mean if with_mean else X.copy()

    if with_std:
        std = Xc.std(axis=0, ddof=ddof)
        scale = std.copy()
        # Avoid division-by-zero: constant features become 0 after centering.
        scale[scale == 0.0] = 1.0
        X_out = Xc / scale
    else:
        scale = np.ones(X.shape[1], dtype=float)
        X_out = Xc

    if return_params:
        return X_out, {"mean": mean, "scale": scale}
    return X_out


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Feature-wise min-max scaling into a target range.

    Each column is mapped from [min, max] into [range_min, range_max].
    If a feature has zero range (min == max), it is mapped entirely to
    range_min (implemented by using scale=1.0).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    feature_range : tuple(float, float), default=(0.0, 1.0)
        Desired output range (min, max).
    return_params : bool, default=False
        If True, also return a dict describing the fitted scaling params.

    Returns
    -------
    X_out : ndarray, shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only when return_params=True. Contains 'min', 'scale', and 'feature_range'.

    Raises
    ------
    ValueError
        If feature_range is invalid or X is malformed.
    TypeError
        If X cannot be coerced to numeric floats.
    """
    X = _ensure_2d_numeric(X, "X")

    if (
        not isinstance(feature_range, tuple)
        or len(feature_range) != 2
        or not all(isinstance(v, (int, float)) for v in feature_range)
    ):
        raise ValueError("feature_range must be a tuple of two numeric values (min, max).")

    fr_min, fr_max = float(feature_range[0]), float(feature_range[1])
    if fr_min >= fr_max:
        raise ValueError("feature_range must have min < max.")

    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    range_ = Xmax - Xmin

    scale = range_.copy()
    scale[scale == 0.0] = 1.0  # constant columns map to fr_min

    X01 = (X - Xmin) / scale
    X_out = X01 * (fr_max - fr_min) + fr_min

    if return_params:
        return X_out, {"min": Xmin, "scale": scale, "feature_range": (fr_min, fr_max)}
    return X_out


def maxabs_scale(
    X: ArrayLike,
    *,
    return_params: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Feature-wise scaling by maximum absolute value.

    Each feature column is divided by max(abs(col)). Constant-zero columns
    remain unchanged (scale defaults to 1.0).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    return_params : bool, default=False
        If True, also return a dict containing 'scale'.

    Returns
    -------
    X_out : ndarray, shape (n_samples, n_features)
        Scaled array.
    params : dict, optional
        Only when return_params=True.
    """
    X = _ensure_2d_numeric(X, "X")

    maxabs = np.max(np.abs(X), axis=0)
    scale = maxabs.copy()
    scale[scale == 0.0] = 1.0

    X_out = X / scale

    if return_params:
        return X_out, {"scale": scale}
    return X_out


def l2_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization.

    Each row x is normalized as:
        x / max(||x||_2, eps)

    This keeps zero rows as zero while preventing division-by-zero.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Input matrix.
    eps : float, default=1e-12
        Small positive constant used as a floor for row norms.

    Returns
    -------
    X_out : ndarray, shape (n_samples, n_features)
        L2-normalized rows.

    Raises
    ------
    ValueError
        If eps <= 0.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    X = _ensure_2d_numeric(X, "X")

    norms = np.linalg.norm(X, axis=1)
    denom = np.maximum(norms, eps)[:, None]
    return X / denom


# ---------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------
def _stratified_indices(y: np.ndarray, test_size: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a stratified train/test split (indices only).

    Notes
    -----
    - Ensures each class contributes proportionally.
    - For very small class sizes, keeps train non-empty when possible.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    train_idx_parts: list[np.ndarray] = []
    test_idx_parts: list[np.ndarray] = []

    for cls in range(len(classes)):
        cls_indices = np.flatnonzero(y_indices == cls)
        rng.shuffle(cls_indices)

        # Try to keep both train and test non-empty when possible.
        n_test = int(round(test_size * len(cls_indices)))
        if len(cls_indices) > 1:
            n_test = min(max(n_test, 1), len(cls_indices) - 1)

        test_idx_parts.append(cls_indices[:n_test])
        train_idx_parts.append(cls_indices[n_test:])

    return np.concatenate(train_idx_parts), np.concatenate(test_idx_parts)


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Split a dataset into train and test subsets.

    If `y` is provided, it is split using the same indices as `X`.
    If `stratify` is provided, the split preserves class proportions
    (and `shuffle` is ignored).

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Targets/labels.
    test_size : float, default=0.2
        Fraction of samples to allocate to the test set.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    stratify : array_like, optional
        Labels to use for stratified splitting.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    (X_train, X_test) or (X_train, X_test, y_train, y_test)

    Raises
    ------
    ValueError
        If sizes are invalid or shapes mismatch.
    TypeError
        If random_state is not an int or None.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(30, dtype=float).reshape(15, 2)
    >>> y = np.array([0]*10 + [1]*5)
    >>> Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
    >>> (yte.tolist().count(0), yte.tolist().count(1))
    (2, 1)
    """
    X = _ensure_2d_numeric(X, "X")
    y_arr = _ensure_1d_vector(y, "y")
    _check_Xy_shapes(X, y_arr)

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float in (0, 1).")

    n = X.shape[0]
    rng = _rng_from_seed(random_state)

    if stratify is not None:
        strat = _ensure_1d_vector(stratify, "stratify")
        if len(strat) != n:
            raise ValueError("stratify must have the same length as X.")
        train_idx, test_idx = _stratified_indices(strat, test_size, rng)
    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        n_test = int(round(test_size * n))
        if n > 1:
            n_test = min(max(n_test, 1), n - 1)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]

    if y_arr is None:
        return X_train, X_test

    y_train, y_test = y_arr[train_idx], y_arr[test_idx]
    return X_train, X_test, y_train, y_test


def train_val_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    val_size: float = 0.1,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Split a dataset into train/validation/test subsets.

    The split is performed as:
    1) carve out the test set (size = test_size)
    2) split the remaining data into train and validation according to val_size

    If `stratify` is provided, each class is split proportionally across
    train/val/test.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Feature matrix.
    y : array_like, shape (n_samples,), optional
        Targets/labels.
    val_size : float, default=0.1
        Fraction of samples allocated to validation.
    test_size : float, default=0.2
        Fraction of samples allocated to test.
    shuffle : bool, default=True
        Whether to shuffle before splitting (ignored for stratified splitting).
    stratify : array_like, optional
        Labels used for stratification.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    (X_train, X_val, X_test) or (X_train, X_val, X_test, y_train, y_val, y_test)

    Raises
    ------
    ValueError
        If split fractions are invalid or shapes mismatch.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(60, dtype=float).reshape(30, 2)
    >>> y = np.array([0]*15 + [1]*15)
    >>> out = train_val_test_split(X, y, val_size=0.2, test_size=0.2, stratify=y, random_state=7)
    >>> Xtr, Xv, Xte, ytr, yv, yte = out
    >>> (len(ytr), len(yv), len(yte))
    (18, 6, 6)
    """
    if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be floats in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0.")

    X = _ensure_2d_numeric(X, "X")
    y_arr = _ensure_1d_vector(y, "y")
    _check_Xy_shapes(X, y_arr)

    n = X.shape[0]
    rng = _rng_from_seed(random_state)

    # Convert val_size into “fraction of remaining after test split”.
    val_prop_remaining = val_size / (1.0 - test_size)

    def _bounded_count(k: int, prop: float) -> int:
        """Round prop*k but try to keep both sides non-empty when k>1."""
        c = int(round(prop * k))
        if k <= 1:
            return 0
        return min(max(c, 1), k - 1)

    if stratify is not None:
        strat = _ensure_1d_vector(stratify, "stratify")
        if len(strat) != n:
            raise ValueError("stratify must have the same length as X.")

        classes, y_idx = np.unique(strat, return_inverse=True)
        train_parts, val_parts, test_parts = [], [], []

        for cls in range(len(classes)):
            cls_indices = np.flatnonzero(y_idx == cls)
            rng.shuffle(cls_indices)

            n_test_c = _bounded_count(len(cls_indices), test_size)
            test_c = cls_indices[:n_test_c]
            remaining_c = cls_indices[n_test_c:]

            n_val_c = _bounded_count(len(remaining_c), val_prop_remaining) if len(remaining_c) > 1 else 0
            val_c = remaining_c[:n_val_c]
            train_c = remaining_c[n_val_c:]

            test_parts.append(test_c)
            val_parts.append(val_c)
            train_parts.append(train_c)

        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

        # Final shuffle within each split to avoid class-block ordering.
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    else:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)

        n_test = _bounded_count(n, test_size)
        test_idx = indices[:n_test]
        remaining = indices[n_test:]

        n_val = _bounded_count(len(remaining), val_prop_remaining) if len(remaining) > 1 else 0
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    if y_arr is None:
        return X_train, X_val, X_test

    y_train, y_val, y_test = y_arr[train_idx], y_arr[val_idx], y_arr[test_idx]
    return X_train, X_val, X_test, y_train, y_val, y_test
