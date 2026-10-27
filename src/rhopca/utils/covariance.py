"""
Kernel and covariance utilities for rhoPCA classes.

All covariance computation lives here.  Column means are computed internally
and centering is applied via algebraic expansion — sparse matrices are never
densified.

Kernel-weighted covariances are built in two steps:

1. :func:`build_weight_matrix` — builds the (n, n) weight matrix ``W`` from
   coordinates or distances.  If ``kernel_kwargs`` contains ``support``, the
   kernel is exactly zero beyond that distance and ``W`` is built sparse from
   a radius-neighbor search; otherwise ``W`` is dense (all pairs).
2. :func:`weighted_covariance` — computes ``(X - 1μᵀ)ᵀ W (X - 1μᵀ) / c``.

Dispatch pattern in :func:`weighted_covariance`:

* ``W=None``,   sparse X  →  :func:`_cov_sparse_standard`
* ``W=None``,   dense X   →  :func:`_cov_dense_standard`
* sparse ``W``            →  :func:`_cov_sparse_kernel`  (scipy.sparse ops)
* dense ``W``             →  :func:`_cov_dense_kernel`   (Numba)

Normalization ``c``: ``n - 1`` (``bias=False``) or ``n`` (``bias=True``), so
that ``W = I`` recovers the standard covariance.  For kernel-weighted
covariances, ``normalize_by_weights=True`` instead divides by
``S0 = Σ_ij W_ij`` (as in Moran's I), controlling for the total weight in ``W``.
"""

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist as _pdist
from numba import njit

from .misc import squareform_from_condensed


# kernel functions ──────────────────────────────────────────────────────────

_BUILTIN_KERNELS = frozenset({'gaussian', 'inverse_distance', 'wendland'})

# Above this many observations a dense (n, n) W is not built; a support is
# chosen automatically (Gaussian) or required (other kernels).
_MAX_DENSE_OBS = 100_000

# Automatic support for large datasets, as a multiple of the Gaussian bandwidth.
# The Wendland taper then shrinks the effective bandwidth by only ~9%.
_AUTO_SUPPORT_FACTOR = 10.0

# Refuse to build a sparse W with more stored entries than this (~12 GB).
_MAX_SPARSE_NNZ = 1_000_000_000

# Number of points subsampled to estimate the median distance and the
# number of neighbors for large datasets.
_N_SUBSAMPLE = 2000


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

    The kernel value at zero distance is ``1 / buffer``, which is used as the
    diagonal of ``W``.  ``buffer`` should therefore be on the scale of the
    distances; a very small buffer makes ``W`` close to ``(1 / buffer) · I``.

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


@njit
def wendland_kernel(distances, support=1.0):
    """
    Compactly supported Wendland kernel ``(1 - r)₊⁴ (4r + 1)`` with
    ``r = d / support``.

    This is the C² Wendland function φ₃,₁, which is positive semi-definite in
    up to 3 dimensions and exactly zero for ``d >= support``.  It is used both
    as a standalone kernel and as a taper for other kernels.

    Parameters
    ----------
    distances : 1-D float64 array
    support : float
        Support radius; weights vanish beyond this distance.

    Returns
    -------
    weights : 1-D float64 array, same length as *distances*
    """
    out = np.zeros_like(distances)
    for i in range(distances.shape[0]):
        r = distances[i] / support
        if r < 1.0:
            t = 1.0 - r
            out[i] = t ** 4 * (4.0 * r + 1.0)
    return out


def apply_kernel(distances, kernel, kernel_kwargs=None):
    """
    Apply a kernel function to a vector of distances.

    Parameters
    ----------
    distances : 1-D array-like
        Pairwise distances.
    kernel : str or callable
        ``'gaussian'``         — Gaussian kernel; accepts ``bandwidth`` kwarg
                                 (default: median of *distances*).
        ``'inverse_distance'`` — inverse-distance kernel; accepts ``buffer`` kwarg.
        ``'wendland'``         — compactly supported Wendland kernel; requires
                                 ``support`` kwarg.
        *callable*             — any function with signature
                                 ``f(distances: ndarray, **kwargs) -> ndarray``.
    kernel_kwargs : dict, optional
        Passed to the kernel function.  For any kernel other than
        ``'wendland'``, a ``support`` entry tapers the kernel entrywise by a
        Wendland kernel with that support, so weights are exactly zero beyond
        ``support`` while the product stays PSD.  ``support`` is not passed
        to callables.

    Returns
    -------
    weights : 1-D float64 array
    """
    kw = dict(kernel_kwargs or {})
    distances = np.ascontiguousarray(distances, dtype=np.float64)

    if kernel == 'wendland':
        if 'support' not in kw:
            raise ValueError("kernel='wendland' requires a 'support' entry in kernel_kwargs.")
        return wendland_kernel(distances, float(kw['support']))

    support = kw.pop('support', None)

    if callable(kernel):
        weights = np.asarray(kernel(distances, **kw), dtype=np.float64)
    elif kernel == 'gaussian':
        bandwidth = kw['bandwidth'] if 'bandwidth' in kw else np.median(distances)
        weights = gaussian_kernel(distances, bandwidth)
    elif kernel == 'inverse_distance':
        weights = inverse_distance_kernel(distances, kw.get('buffer', 1e-6))
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Built-in options: {sorted(_BUILTIN_KERNELS)}. "
            "Or pass a callable with signature f(distances, **kwargs) -> weights."
        )

    if support is not None:
        weights = weights * wendland_kernel(distances, float(support))
    return weights


def _kernel_at_zero(kernel, kernel_kwargs=None):
    """Kernel value at zero distance, used as the diagonal of ``W``."""
    kw = kernel_kwargs or {}
    if kernel in ('gaussian', 'wendland'):
        return 1.0
    if kernel == 'inverse_distance':
        return 1.0 / kw.get('buffer', 1e-6)
    return float(apply_kernel(np.zeros(1), kernel, kw)[0])


def _warn_if_diagonal_dominates(diag_value, offdiag_weights):
    """Warn when ``W`` is effectively a scaled identity matrix."""
    max_off = np.max(offdiag_weights) if len(offdiag_weights) else 0.0
    if diag_value > 1e3 * max_off:
        warnings.warn(
            f"Kernel value at zero distance ({diag_value:.3g}) is more than 1000x the "
            f"largest off-diagonal weight ({max_off:.3g}), so W is close to a scaled "
            "identity and the kernel has little effect.  For 'inverse_distance', "
            "set kernel_kwargs['buffer'] on the scale of the distances."
        )


def _subsample(coords, rng=None):
    """Random subsample of at most ``_N_SUBSAMPLE`` rows."""
    n = coords.shape[0]
    if n <= _N_SUBSAMPLE:
        return coords
    rng = np.random.default_rng(0) if rng is None else rng
    return coords[rng.choice(n, _N_SUBSAMPLE, replace=False)]


def _check_sparse_size(coords, support):
    """Estimate the stored entries of a sparse W and refuse if too large."""
    tree = cKDTree(coords)
    counts = tree.query_ball_point(_subsample(coords), r=support, return_length=True)
    est_nnz = float(np.mean(counts)) * coords.shape[0]
    if est_nnz > _MAX_SPARSE_NNZ:
        raise ValueError(
            f"support={support:.4g} gives about {np.mean(counts):.0f} neighbors per "
            f"observation (~{est_nnz:.2g} stored weights, ~{est_nnz * 12 / 1e9:.0f} GB). "
            "Use a smaller kernel_kwargs['support'] (or a smaller 'bandwidth' for "
            "'gaussian', which sets the automatic support)."
        )


def build_weight_matrix(
    n_obs,
    *,
    kernel,
    coordinates=None,
    distances=None,
    kernel_kwargs=None,
):
    """
    Build the symmetric (n, n) kernel weight matrix ``W``.

    Distances are Euclidean.  The diagonal is set to the kernel value at zero
    distance, so ``W`` is positive semi-definite whenever the kernel is.

    **Sparse vs dense.**  If ``kernel_kwargs`` contains ``support`` (required
    for ``'wendland'``), weights are exactly zero beyond that distance, so
    ``W`` is built sparse from a radius-neighbor search with
    ``radius = support``.  For other kernels, ``support`` tapers the kernel
    by a Wendland kernel (``W = K ∘ K_Wendland``), which keeps ``W`` PSD; a
    hard cutoff alone would not.  Without ``support``, ``W`` is dense.

    **Large datasets.**  With ``n_obs >= 100,000`` a dense ``W`` does not fit
    in memory, so a missing ``support`` is chosen automatically for
    ``'gaussian'`` as ``10 × bandwidth`` (with a warning); other kernels
    require ``support``.  If the support would give too many neighbors, a
    ``ValueError`` is raised.  Note that the default Gaussian bandwidth (the
    median pairwise distance) is on the scale of the whole dataset, so large
    datasets need an explicit, local ``bandwidth``.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    kernel, coordinates, distances, kernel_kwargs
        See :func:`compute_covariance`.

    Returns
    -------
    W : (n, n) float64 ndarray or scipy.sparse CSR matrix
    """
    if coordinates is None and distances is None:
        raise ValueError("Supply `coordinates` or `distances` when using a kernel.")

    kw = dict(kernel_kwargs or {})
    coords = None if coordinates is None else np.asarray(coordinates, dtype=np.float64)

    if coords is not None and coords.shape[1] > 3 and (kernel == 'wendland' or 'support' in kw):
        warnings.warn(
            f"Coordinates have {coords.shape[1]} dimensions; the Wendland kernel "
            "(standalone or as a taper) is only guaranteed PSD in up to 3 dimensions."
        )

    # dense path from precomputed distances
    if distances is not None:
        weights = apply_kernel(distances, kernel, kernel_kwargs=kw)
        W = squareform_from_condensed(np.ascontiguousarray(weights, dtype=np.float64))
        diag_value = _kernel_at_zero(kernel, kw)
        _warn_if_diagonal_dominates(diag_value, weights)
        np.fill_diagonal(W, diag_value)
        return np.ascontiguousarray(W, dtype=np.float64)

    large = n_obs >= _MAX_DENSE_OBS

    # Resolve the default Gaussian bandwidth over all pairs (subsampled when
    # large), not just the pairs within the support.
    if kernel == 'gaussian' and 'bandwidth' not in kw and ('support' in kw or large):
        kw['bandwidth'] = float(np.median(_pdist(_subsample(coords))))

    if 'support' not in kw and large:
        if kernel != 'gaussian':
            raise ValueError(
                f"{n_obs} observations are too many for a dense weight matrix. "
                "Pass kernel_kwargs['support'] (the distance beyond which weights are "
                "zero) to build a sparse one."
            )
        kw['support'] = _AUTO_SUPPORT_FACTOR * kw['bandwidth']
        warnings.warn(
            f"{n_obs} observations are too many for a dense weight matrix; using "
            f"support = {_AUTO_SUPPORT_FACTOR:g} x bandwidth = {kw['support']:.4g} "
            f"(bandwidth = {kw['bandwidth']:.4g}).  Set kernel_kwargs['support'] "
            "to choose it explicitly."
        )

    # sparse path: only pairs within the support
    if 'support' in kw:
        from sklearn.neighbors import radius_neighbors_graph

        support = float(kw['support'])
        _check_sparse_size(coords, support)
        W = radius_neighbors_graph(coords, radius=support, mode='distance').tocsr()
        W.data = apply_kernel(W.data, kernel, kernel_kwargs=kw)
        W.eliminate_zeros()

        diag_value = _kernel_at_zero(kernel, kw)
        _warn_if_diagonal_dominates(diag_value, W.data)
        return (W + scipy.sparse.diags(np.full(n_obs, diag_value))).tocsr()

    # dense path: all pairs
    distances = _pdist(coords)
    weights = apply_kernel(distances, kernel, kernel_kwargs=kw)
    W = squareform_from_condensed(np.ascontiguousarray(weights, dtype=np.float64))
    diag_value = _kernel_at_zero(kernel, kw)
    _warn_if_diagonal_dominates(diag_value, weights)
    np.fill_diagonal(W, diag_value)
    return np.ascontiguousarray(W, dtype=np.float64)


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

    return np.atleast_2d(np.cov(X.T, bias=bias))


def _cov_sparse_kernel(X, W, denom):
    """
    Kernel-weighted covariance for sparse W (X dense or sparse).

    All multiplications use scipy.sparse dot products so that neither X nor W
    is explicitly densified.

    Parameters
    ----------
    X : (n, p) sparse matrix or float64 ndarray
    W : sparse (n, n)
        Symmetric weight matrix.
    denom : float
        Normalization constant (n-1, n, or S0).

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    w = np.asarray(W.sum(axis=1), dtype=np.float64).ravel()   # (n,)
    Xw = np.asarray(X.T @ w, dtype=np.float64).ravel()          # (p,)
    S_W = w.sum()

    XtWX = X.T @ (W @ X)                                        # (p, p)
    if scipy.sparse.issparse(XtWX):
        XtWX = XtWX.toarray()
    XtWX = np.asarray(XtWX, dtype=np.float64)

    correction = np.outer(Xw, mu) + np.outer(mu, Xw) - S_W * np.outer(mu, mu)
    return (XtWX - correction) / denom


@njit
def _cov_dense_kernel(X, W, denom):
    """
    Kernel-weighted covariance for dense X and W (Numba-accelerated).

    Parameters
    ----------
    X : (n, p) float64 ndarray
    W : (n, n) float64 ndarray
    denom : float
        Normalization constant (n-1, n, or S0).

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    n = X.shape[0]
    mu = X.sum(axis=0) / n           # (p,) column means
    w = W.sum(axis=1)                # (n,) row sums
    Xw = X.T @ w                     # (p,)
    S_W = w.sum()
    XtWX = X.T @ (W @ X)            # (p, p)
    correction = np.outer(Xw, mu) + np.outer(mu, Xw) - S_W * np.outer(mu, mu)
    return (XtWX - correction) / denom


def _warn_if_not_psd(cov):
    """Warn if a weighted covariance has materially negative eigenvalues."""
    if cov.shape[0] == 0:
        return
    lam_min = scipy.linalg.eigvalsh(cov, subset_by_index=[0, 0])[0]
    scale = max(abs(np.trace(cov)), np.finfo(np.float64).tiny)
    if lam_min < -1e-8 * scale:
        warnings.warn(
            f"Kernel-weighted covariance is not positive semi-definite "
            f"(smallest eigenvalue {lam_min:.3g}).  The weight matrix W is likely "
            "not PSD; use a PSD kernel such as 'gaussian' or 'wendland'."
        )


def weighted_covariance(X, W=None, *, bias=False, normalize_by_weights=False, check_psd=True):
    """
    Compute ``(X - 1μᵀ)ᵀ W (X - 1μᵀ) / c`` for a given weight matrix.

    Parameters
    ----------
    X : (n, p) array-like — dense ndarray or scipy.sparse matrix
        Raw or scaled data matrix.  Centering is handled internally.
    W : (n, n) ndarray, scipy.sparse matrix, or None
        Symmetric weight matrix.  ``None`` is equivalent to the identity and
        returns the standard covariance.
    bias : bool, default False
        ``False`` divides by n-1 (sample covariance); ``True`` divides by n.
        With ``W = I`` this recovers ``numpy.cov(X.T, bias=bias)``.
    normalize_by_weights : bool, default False
        Only used when *W* is given.  Divide by ``S0 = Σ_ij W_ij`` (as in
        Moran's I) instead of n-1 / n, controlling for the total weight in
        *W*.  Overrides *bias*.
    check_psd : bool, default True
        Warn if the weighted covariance has materially negative eigenvalues.
        Costs one smallest-eigenvalue computation on the (p, p) result.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    X_is_sparse = scipy.sparse.issparse(X)

    if W is None:
        if X_is_sparse:
            return _cov_sparse_standard(X, bias=bias)
        return _cov_dense_standard(np.asarray(X, dtype=np.float64), bias=bias)

    n = X.shape[0]
    if normalize_by_weights:
        denom = float(W.sum())
        if denom <= 0:
            raise ValueError(f"Sum of weights S0 = {denom:.3g} must be positive.")
    else:
        denom = float(n if bias else n - 1)

    if scipy.sparse.issparse(W):
        if not X_is_sparse:
            X = np.asarray(X, dtype=np.float64)
        cov = _cov_sparse_kernel(X, W, denom)
    else:
        # W is dense; densify X if needed and use the Numba path
        if X_is_sparse:
            X = X.toarray()
        cov = _cov_dense_kernel(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W, dtype=np.float64),
            denom,
        )

    if check_psd:
        _warn_if_not_psd(cov)
    return cov


def compute_covariance(
    X,
    *,
    kernel=None,
    bias=False,
    normalize_by_weights=False,
    coordinates=None,
    distances=None,
    kernel_kwargs=None,
    return_weights=False,
):
    """
    Compute a standard or kernel-weighted covariance matrix.

    Parameters
    ----------
    X : (n, p) array-like — dense ndarray or scipy.sparse matrix
        Raw or scaled data matrix.  Centering is handled internally.
    kernel : str, callable, or None
        Kernel to apply.  ``None`` returns the standard covariance.
        See :func:`apply_kernel`.  The kernel should be positive
        semi-definite so that the weighted covariance is PSD.
    bias : bool, default False
        Normalization convention.  ``False`` divides by n-1 (sample
        covariance); ``True`` divides by n (population covariance),
        matching ``numpy.cov(bias=True)``.  With ``W = I`` the kernel-weighted
        covariance reduces to the standard one.
    normalize_by_weights : bool, default False
        Only used when a kernel is applied.  Divide by ``S0 = Σ_ij W_ij``
        (as in Moran's I) instead of n-1 / n, controlling for the total
        weight in ``W``.  Overrides *bias* for the kernel-weighted covariance.
    coordinates : (n, d) array-like, optional
        Spatial or other coordinates; Euclidean distances between them are
        passed to the kernel.
    distances : 1-D array-like, optional
        Pre-computed condensed distance vector (e.g. from
        ``scipy.spatial.distance.pdist``).  Takes priority over
        *coordinates* and always gives a dense ``W``.
    kernel_kwargs : dict, optional
        Kernel parameters:

        * ``bandwidth`` — ``'gaussian'`` width σ: how quickly weights decay.
          Defaults to the median pairwise distance.
        * ``buffer`` — ``'inverse_distance'`` offset in ``1 / (d + buffer)``.
        * ``support`` — distance beyond which weights are exactly zero.
          Required for ``'wendland'``; optional for other kernels, which are
          then tapered by a Wendland kernel.  When given, ``W`` is built
          sparse from the pairs within ``support``.  A support of about
          10 × bandwidth keeps a Gaussian close to its untapered shape.

        See :func:`build_weight_matrix` for how large datasets are handled.
    return_weights : bool, default False
        If True, return ``(cov, W)``; ``W`` is ``None`` when no kernel is used.

    Returns
    -------
    cov : (p, p) float64 ndarray
    W : (n, n) ndarray, sparse matrix, or None
        Only returned when ``return_weights=True``.
    """
    if kernel is None:
        W = None
    else:
        W = build_weight_matrix(
            X.shape[0],
            kernel=kernel,
            coordinates=coordinates,
            distances=distances,
            kernel_kwargs=kernel_kwargs,
        )

    cov = weighted_covariance(X, W, bias=bias, normalize_by_weights=normalize_by_weights)
    return (cov, W) if return_weights else cov
