"""
Kernel and covariance utilities for rhoPCA classes.

All covariance computation lives here.  Column means are computed internally
and centering is applied via algebraic expansion — sparse matrices are never
densified.

Dispatch pattern:

* ``kernel=None``, sparse  →  :func:`_cov_sparse_standard`
* ``kernel=None``, dense   →  direct matrix product
* ``kernel``,     sparse   →  :func:`_cov_sparse_kernel`  (scipy.sparse ops)
* ``kernel``,     dense    →  :func:`_cov_dense_kernel`   (Numba) + numpy correction
"""

import numpy as np
import scipy.sparse
from scipy.spatial.distance import pdist as _pdist
from numba import njit

from .misc import squareform_from_condensed


# kernel functions ──────────────────────────────────────────────────────────

_BUILTIN_KERNELS = frozenset({'gaussian', 'inverse_distance'})

@njit
def gaussian_kernel(distances, bandwidth=1.0):
    """
    Gaussian kernel: ``exp(-d² / (2 · bandwidth²))``.

    Parameters
    ----------
    distances : 1-D float64 array
        Pairwise distances (condensed or full).
    bandwidth : float
        Kernel bandwidth sigma.  Larger values imply smoother weighting.

    Returns
    -------
    weights : 1-D float64 array, same length as *distances*
    """
    return np.exp(-np.square(distances) / (2.0 * bandwidth ** 2))


@njit
def inverse_distance_kernel(distances, buffer=1e-6):
    """
    Inverse-distance kernel: ``1 / (d + buffer)``.

    Parameters
    ----------
    distances : 1-D float64 array
    buffer : float
        Small constant added to prevent division by zero.

    Returns
    -------
    weights : 1-D float64 array, same length as *distances*
    """
    return 1.0 / (distances + buffer)


def apply_kernel(distances, kernel, kernel_kwargs=None):
    """
    Apply a kernel function to a condensed distance vector.

    Parameters
    ----------
    distances : 1-D array-like
        Condensed pairwise distances.
    kernel : str or callable
        ``'gaussian'``         — Gaussian kernel; accepts ``bandwidth`` kwarg.
        ``'inverse_distance'`` — inverse-distance kernel; accepts ``buffer`` kwarg.
        *callable*             — any function with signature
                                 ``f(distances: ndarray, **kwargs) -> ndarray``.
    kernel_kwargs : dict, optional
        Passed to the kernel function.

    Returns
    -------
    weights : 1-D float64 array
    """
    kw = kernel_kwargs or {}
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    if callable(kernel):
        return np.asarray(kernel(distances, **kw), dtype=np.float64)

    if kernel == 'gaussian':
        bw = np.sqrt(np.median(distances))
        return gaussian_kernel(distances, kw.get('bandwidth', bw))

    if kernel == 'inverse_distance':
        return inverse_distance_kernel(distances, kw.get('buffer', 1e-6))

    raise ValueError(
        f"Unknown kernel '{kernel}'. "
        f"Built-in options: {sorted(_BUILTIN_KERNELS)}. "
        "Or pass a callable with signature f(distances, **kwargs) -> weights."
    )


# covariance helpers ────────────────────────────────────────────────────────

def _cov_sparse_standard(X, bias=False):
    """
    Standard covariance for sparse X via scipy.sparse ops.

    Parameters
    ----------
    X : sparse (n, p)
    bias : bool, default False
        If True normalize by n; if False normalize by n-1.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    n = X.shape[0]
    dof = n if bias else n - 1
    mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    XtX = X.T.dot(X)

    if scipy.sparse.issparse(XtX):
        XtX = XtX.toarray()
    return (np.asarray(XtX, dtype=np.float64) - n * np.outer(mu, mu)) / dof


def _cov_dense_standard(X, bias=False):
    """
    Standard covariance for dense X.

    Parameters
    ----------
    X : (n, p) float64 ndarray
    bias : bool, default False
        If True normalize by n; if False normalize by n-1.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    # n = X.shape[0]
    # dof = n if bias else n - 1
    # mu = X.mean(axis=0)
    # return (X.T @ X - n * np.outer(mu, mu)) / dof

    return np.cov(X.T, bias=bias)


def _cov_sparse_kernel(X, W, bias=False):
    """
    Kernel-weighted covariance for sparse X.

    All multiplications use scipy.sparse dot products so that X is
    never explicitly densified.

    Parameters
    ----------
    X : sparse (n, p)
    W : (n, n) float64 ndarray
        Symmetric weight matrix.
    bias : bool, default False
        If True normalize by n; if False normalize by n-1.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    
    n = X.shape[0]
    dof = n if bias else n - 1
    mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    w = W.sum(axis=1)                                # (n,)
    Xw = np.asarray(X.T.dot(w)).ravel()              # (p,)
    S_W = w.sum()

    WX = (X.T.dot(W.T)).T                  # (n, p) dense
    # XtWX = np.asarray(X.T.dot(WX))                   # (p, p) dense
    XtWX = (X.T.dot(WX)).toarray()                   # (p, p) dense

    correction = np.outer(Xw, mu) + np.outer(mu, Xw) - S_W * np.outer(mu, mu)
    return (XtWX - correction) / dof


@njit
def _cov_dense_kernel(X, W, bias=False):
    """
    Kernel-weighted covariance for dense X (Numba-accelerated).

    Parameters
    ----------
    X : (n, p) float64 ndarray
    W : (n, n) float64 ndarray
    bias : bool, default False
        If True normalize by n; if False normalize by n-1.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    n = X.shape[0]
    dof = n if bias else n - 1
    mu = X.sum(axis=0) / n           # (p,) column means
    w = W.sum(axis=1)                # (n,) row sums
    Xw = X.T @ w                     # (p,)
    S_W = w.sum()
    XtWX = X.T @ (W @ X)            # (p, p)
    correction = np.outer(Xw, mu) + np.outer(mu, Xw) - S_W * np.outer(mu, mu)
    return (XtWX - correction) / dof


def compute_covariance(
    X,
    *,
    kernel=None,
    bias=False,
    coordinates=None,
    distances=None,
    pdist_kwargs=None,
    radius_neighbors_kwargs=None,
    use_radius_neighbors=True,
    kernel_kwargs=None,
):
    """
    Compute a standard or kernel-weighted covariance matrix.

    Dispatch strategy:

    * **Sparse input** — uses ``scipy.sparse`` native dot products throughout;
      no explicit conversion to dense.
    * **Dense input** — uses the Numba-accelerated :func:`_cov_dense_kernel`
      for the kernel path, or a direct matrix product for the standard path.

    Parameters
    ----------
    X : (n, p) array-like — dense ndarray or scipy.sparse matrix
        Raw or scaled data matrix.  Centering is handled internally.
    kernel : str, callable, or None
        Kernel to apply.  ``None`` returns the standard covariance.
        See :func:`apply_kernel`.
    bias : bool, default False
        Normalization convention.  ``False`` divides by n-1 (sample
        covariance); ``True`` divides by n (population covariance),
        matching ``numpy.cov(bias=True)``.
    coordinates : (n, d) array-like, optional
        Spatial or other coordinates used to compute pairwise distances when
        *distances* is not supplied.
    distances : 1-D array-like, optional
        Pre-computed condensed distance vector.  Takes priority over
        *coordinates*.
    pdist_kwargs : dict, optional
        Keyword arguments forwarded to ``scipy.spatial.distance.pdist`` when
        computing distances from *coordinates*.  Defaults to Euclidean.
        Only used when ``use_radius_neighbors=False`` or n_obs < 100,000.
    radius_neighbors_kwargs : dict, optional
        Keyword arguments forwarded to
        ``sklearn.neighbors.radius_neighbors_graph``.  Defaults to
        ``radius=100, mode='distance'``.  Any keys supplied here override
        the defaults.  Only used when ``use_radius_neighbors=True``.
    use_radius_neighbors : bool, default True
        When ``True``, a sparse weight matrix is built via
        ``sklearn.neighbors.radius_neighbors_graph`` instead of computing
        all pairwise distances with ``scipy.spatial.distance.pdist``.
        The sparse path is triggered automatically when n_obs >= 100,000,
        or immediately when *radius_neighbors_kwargs* is supplied.
        Set to ``False`` to always compute all pairwise distances regardless
        of dataset size.
    kernel_kwargs : dict, optional
        Passed to the kernel function (e.g. ``{'bandwidth': 2.0}``,
        ``{'buffer': 1e-6}``).

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    X_is_sparse = scipy.sparse.issparse(X)

    # standard covariance
    if kernel is None:
        if X_is_sparse:
            return _cov_sparse_standard(X, bias=bias)
        return _cov_dense_standard(np.asarray(X, dtype=np.float64), bias=bias)

    # kernel-weighted covariance
    if coordinates is None and distances is None:
        raise ValueError("Supply `coordinates` or `distances` when using a kernel.")

    use_radius = use_radius_neighbors and (
        (radius_neighbors_kwargs is not None) or (X.shape[0] >= 100_000)
    )

    if use_radius and coordinates is not None:
        from sklearn.neighbors import radius_neighbors_graph

        coords = np.asarray(coordinates, dtype=np.float64)
        rn_kw = {'radius': 500, 'mode': 'distance'}
        rn_kw.update(radius_neighbors_kwargs or {})

        # Returns a sparse graph
        W = radius_neighbors_graph(coords, **rn_kw)
        W.data = apply_kernel(W.data, kernel, kernel_kwargs=kernel_kwargs)

        if kernel == 'gaussian' or kernel == 'inverse_distance':
            W.setdiag(1.0)

    else:
        if distances is None:
            kw = pdist_kwargs or {}
            distances = _pdist(np.asarray(coordinates, dtype=np.float64), **kw)
        weights = apply_kernel(distances, kernel, kernel_kwargs=kernel_kwargs)
        W = squareform_from_condensed(np.ascontiguousarray(weights, dtype=np.float64))
        
        if kernel == 'gaussian' or kernel == 'inverse_distance':
            np.fill_diagonal(W, 1.0)
        W = np.ascontiguousarray(W, dtype=np.float64)

    W_is_sparse = scipy.sparse.issparse(W)
    if W_is_sparse:
        if X_is_sparse:
            return _cov_sparse_kernel(X, W, bias=bias)
        else:
            W = np.ascontiguousarray(W.toarray(), dtype=np.float64)
            return _cov_dense_kernel(X, W) 
    X_is_sparse:
        X = X.toarray()
    return _cov_dense_kernel(
        np.ascontiguousarray(X, dtype=np.float64),
        W, bias=bias
    )
