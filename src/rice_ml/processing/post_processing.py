"""
post_processing.py

This file collects a practical set of evaluation metrics commonly used after
training a model. The APIs are intentionally lightweight and scikit-learn-ish.

Highlights
----------
- Strong shape/type validation (clear errors for mismatched lengths, bad dims)
- Classification metrics for discrete labels
- Probability-based metrics for classifiers (log loss; ROC AUC for binary)
- Regression metrics for numeric targets

Classification
--------------
accuracy_score
precision_score
recall_score
f1_score
confusion_matrix
roc_auc_score   (binary only; uses scores/ranks)
log_loss        (binary or multiclass; expects probabilities)

Regression
----------
mse
rmse
mae
r2_score
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    "mse",
    "rmse",
    "mae",
    "r2_score",
]

ArrayLike = Union[np.ndarray, Sequence]
NumArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
def _as_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """
    Convert an input into a non-empty 1D NumPy array.

    Notes
    -----
    This helper is intentionally strict: metrics in this module are defined
    for 1D vectors of length n_samples.
    """
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as_1d_float(x: NumArrayLike, name: str) -> np.ndarray:
    """
    Convert an input into a numeric 1D float array.

    Raises
    ------
    TypeError if values cannot be coerced to floats.
    """
    arr = _as_1d_array(x, name)
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _check_pair(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate y_true/y_pred as aligned 1D arrays.
    """
    yt = _as_1d_array(y_true, "y_true")
    yp = _as_1d_array(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have same length; got {yt.shape[0]} vs {yp.shape[0]}."
        )
    return yt, yp


def _check_probabilities(
    y_true: ArrayLike, y_prob: ArrayLike
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate probability inputs for log_loss/roc_auc.

    Accepted formats
    ----------------
    Binary:
      - shape (n,): interpreted as P(positive class)
      - shape (n, 2): interpreted as [P(class0), P(class1)]

    Multiclass:
      - shape (n, K): one row per sample, one column per class

    Returns
    -------
    yt : ndarray shape (n,)
        Raw label vector.
    probs : ndarray shape (n, K)
        Probabilities as a 2D float array.
    K : int
        Number of probability columns.
    """
    yt = _as_1d_array(y_true, "y_true")
    probs = np.asarray(y_prob)

    if probs.ndim == 1:
        # Binary shorthand: a single probability per sample (for the "positive" class).
        probs = probs.astype(float)
        if probs.shape[0] != yt.shape[0]:
            raise ValueError("For binary, y_prob with shape (n,) must match len(y_true).")
        probs = np.stack([1.0 - probs, probs], axis=1)
        K = 2
    elif probs.ndim == 2:
        if probs.shape[0] != yt.shape[0]:
            raise ValueError("y_prob must have the same first dimension as y_true.")
        probs = probs.astype(float)
        K = probs.shape[1]
    else:
        raise ValueError("y_prob must be a 1D or 2D array.")

    # Basic numeric sanity: finite and within [0, 1].
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Probabilities must be finite and in [0, 1].")

    return yt, probs, K


# ---------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------
def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Ground-truth labels.
    y_pred : array_like, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    float
        Proportion of correct predictions.

    Examples
    --------
    >>> accuracy_score(["cat", "dog", "dog"], ["cat", "cat", "dog"])
    0.6666666666666666
    """
    yt, yp = _check_pair(y_true, y_pred)
    return float(np.mean(yt == yp))


def _counts_by_class(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[Sequence] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-class TP/FP/FN via an explicit confusion matrix.

    Notes
    -----
    If `labels` is provided, any sample whose label is not in `labels` is ignored.
    This matches the typical convention used in many libraries.
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    L = len(labels)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    t_idx = np.array([label_to_idx.get(lab, -1) for lab in y_true])
    p_idx = np.array([label_to_idx.get(lab, -1) for lab in y_pred])

    cm = np.zeros((L, L), dtype=int)
    for ti, pi in zip(t_idx, p_idx):
        if ti == -1 or pi == -1:
            continue
        cm[ti, pi] += 1

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    return tp, fp, fn, labels, cm


def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None,
) -> float:
    """
    Precision = TP / (TP + FP).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        - "binary": positive class is max of the two observed labels
        - "macro": mean of per-class precision
        - "micro": compute global TP/(TP+FP)
        - None: return per-class precision (aligned to `labels` or inferred labels)
    labels : sequence, optional
        Label ordering / subset to evaluate.

    Returns
    -------
    float or ndarray
        Precision per the averaging strategy.

    Examples
    --------
    >>> precision_score([1, 1, 0, 0], [1, 0, 0, 0], average="binary")
    1.0
    """
    yt, yp = _check_pair(y_true, y_pred)

    if average == "binary":
        uniq = np.unique(np.concatenate([yt, yp]))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly 2 classes.")
        pos = np.max(uniq)
        tp = np.sum((yt == pos) & (yp == pos))
        fp = np.sum((yt != pos) & (yp == pos))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    tp, fp, _, _, _ = _counts_by_class(yt, yp, labels)

    if average == "micro":
        TP = tp.sum()
        FP = fp.sum()
        return float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)

    if average == "macro":
        return float(np.mean(per_class))
    if average is None:
        return per_class

    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None,
) -> float:
    """
    Recall = TP / (TP + FN).

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        See `precision_score`.
    labels : sequence, optional
        Label ordering / subset to evaluate.

    Returns
    -------
    float or ndarray
        Recall per the averaging strategy.

    Examples
    --------
    >>> recall_score([1, 1, 0, 0], [1, 0, 0, 0], average="binary")
    0.5
    """
    yt, yp = _check_pair(y_true, y_pred)

    if average == "binary":
        uniq = np.unique(np.concatenate([yt, yp]))
        if uniq.size != 2:
            raise ValueError("binary average requires exactly 2 classes.")
        pos = np.max(uniq)
        tp = np.sum((yt == pos) & (yp == pos))
        fn = np.sum((yt == pos) & (yp != pos))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    tp, _, fn, _, _ = _counts_by_class(yt, yp, labels)

    if average == "micro":
        TP = tp.sum()
        FN = fn.sum()
        return float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)

    if average == "macro":
        return float(np.mean(per_class))
    if average is None:
        return per_class

    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: Optional[str] = "binary",
    labels: Optional[Sequence] = None,
) -> float:
    """
    F1 score: harmonic mean of precision and recall.

    F1 = 2PR / (P + R), with the convention F1=0 when P+R=0.

    Parameters
    ----------
    y_true, y_pred : array_like
        True and predicted labels.
    average : {"binary", "macro", "micro", None}, default="binary"
        See `precision_score`.
    labels : sequence, optional
        Label ordering / subset to evaluate.

    Returns
    -------
    float or ndarray
        F1 score per the averaging strategy.

    Examples
    --------
    >>> f1_score([1, 0, 1, 0], [1, 0, 0, 0], average="binary")
    0.6666666666666666
    """
    yt, yp = _check_pair(y_true, y_pred)

    if average in ("binary", "micro"):
        p = precision_score(yt, yp, average=average)
        r = recall_score(yt, yp, average=average)
        return float(0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r))

    tp, fp, fn, _, _ = _counts_by_class(yt, yp, labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        f1 = np.where((prec + rec) > 0, 2.0 * prec * rec / (prec + rec), 0.0)

    if average == "macro":
        return float(np.mean(f1))
    if average is None:
        return f1

    raise ValueError('average must be one of {"binary", "macro", "micro", None}.')


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, *, labels: Optional[Sequence] = None
) -> np.ndarray:
    """
    Build a confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        True and predicted labels.
    labels : sequence, optional
        Explicit class order. If provided, labels not in this list are ignored.

    Returns
    -------
    ndarray, shape (n_classes, n_classes)
        Rows correspond to true classes, columns to predicted classes.

    Examples
    --------
    >>> confusion_matrix(["spam", "ham", "spam"], ["spam", "spam", "ham"]).tolist()
    [[1, 1], [1, 0]]
    """
    yt, yp = _check_pair(y_true, y_pred)
    _, _, _, _, cm = _counts_by_class(yt, yp, labels)
    return cm


def roc_auc_score(y_true: ArrayLike, y_scores: ArrayLike) -> float:
    """
    Compute ROC AUC for **binary** classification.

    This expects a continuous score for the positive class (e.g., probability
    of the positive class, logits, or any rankable score where higher means
    "more positive").

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels containing exactly 2 unique values.
    y_scores : array_like, shape (n_samples,)
        Score for the positive class.

    Returns
    -------
    float
        AUC in [0, 1].

    Examples
    --------
    >>> yt = [0, 1, 0, 1, 1]
    >>> s  = [0.2, 0.9, 0.4, 0.7, 0.6]
    >>> round(roc_auc_score(yt, s), 3)
    1.0
    """
    yt = _as_1d_array(y_true, "y_true")
    ys = _as_1d_float(y_scores, "y_scores")

    uniq = np.unique(yt)
    if uniq.size != 2:
        raise ValueError("roc_auc_score requires exactly 2 classes.")
    if np.all(yt == uniq[0]) or np.all(yt == uniq[1]):
        raise ValueError("y_true must contain at least one sample from each class.")

    # Rank-based AUC (Mann–Whitney U). Stable sort to behave deterministically on ties.
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(ys) + 1, dtype=float)  # ranks 1..n

    pos_label = np.max(uniq)
    pos_mask = yt == pos_label
    n_pos = int(np.sum(pos_mask))
    n_neg = int(len(yt) - n_pos)

    sum_pos_ranks = float(np.sum(ranks[pos_mask]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def log_loss(y_true: ArrayLike, y_prob: ArrayLike, eps: float = 1e-15) -> float:
    """
    Cross-entropy / log loss for binary or multiclass classification.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        True labels.
    y_prob : array_like
        Predicted probabilities:
          - binary: shape (n,) means P(positive), or shape (n, 2) means [P0, P1]
          - multiclass: shape (n, K) with one probability per class
    eps : float, default=1e-15
        Small lower bound used to avoid log(0). (Values are clipped to [eps, 1].)

    Returns
    -------
    float
        Mean negative log-likelihood.

    Raises
    ------
    ValueError
        For invalid shapes, invalid probability ranges, invalid row sums,
        or if labels cannot be mapped to columns.

    Examples
    --------
    Binary (vector of P(positive)):

    >>> y = [0, 1, 1, 0]
    >>> p = [0.1, 0.8, 0.6, 0.2]
    >>> round(log_loss(y, p), 6)
    0.233566
    """
    yt, probs, K = _check_probabilities(y_true, y_prob)

    if eps <= 0 or not np.isfinite(eps):
        raise ValueError("eps must be positive and finite.")

    # If y_prob is 2D, enforce that each row is (approximately) a probability simplex.
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("Each probability row must sum to 1 within tolerance.")

    labels = np.unique(yt)

    # Column mapping:
    # - For binary and exactly two labels, map min->0, max->1
    # - Otherwise, attempt to align unique labels with K if possible; fallback to 0..K-1
    if K == 2 and labels.size == 2:
        label_to_col = {labels.min(): 0, labels.max(): 1}
    else:
        if labels.size != K:
            labels = np.arange(K)
        label_to_col = {lab: i for i, lab in enumerate(labels)}

    cols = np.array([label_to_col.get(lab, -1) for lab in yt], dtype=int)
    if np.any(cols < 0):
        raise ValueError("Could not map some y_true labels to probability columns.")

    # Clip only on the bottom; allow exact 1.0 for perfect predictions.
    p = np.clip(probs, eps, 1.0)
    losses = -np.log(p[np.arange(len(yt)), cols])
    return float(np.mean(losses))


# ---------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------
def mse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean squared error.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        Numeric targets and predictions.

    Returns
    -------
    float
        Mean((y_true - y_pred)^2)

    Examples
    --------
    >>> mse([0, 1, 2], [0, 2, 1])
    0.6666666666666666
    """
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Root mean squared error.

    Examples
    --------
    >>> round(rmse([0, 1, 2], [0, 2, 1]), 6)
    0.816497
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    Mean absolute error.

    Examples
    --------
    >>> mae([0, 1, 2], [0, 2, 1])
    0.6666666666666666
    """
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: NumArrayLike, y_pred: NumArrayLike) -> float:
    """
    R^2 (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot, where SS_tot is computed around mean(y_true).

    Behavior for constant y_true
    ----------------------------
    If y_true is constant:
      - returns 1.0 if predictions are perfect
      - otherwise raises ValueError (R^2 is not meaningful in that case)

    Examples
    --------
    >>> r2_score([1, 2, 3], [1, 2, 3])
    1.0
    >>> round(r2_score([1, 2, 3], [1, 2, 4]), 6)
    0.5
    """
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have same length.")

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))

    if ss_tot == 0.0:
        if ss_res == 0.0:
            return 1.0
        raise ValueError("R^2 is undefined when y_true is constant and predictions are not perfect.")

    return float(1.0 - ss_res / ss_tot)
