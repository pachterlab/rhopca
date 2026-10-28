"""
Kernel and covariance utilities for rhoPCA classes.

All covariance computation lives here.  Column means are computed internally
and centering is applied via algebraic expansion — sparse matrices are never
densified as a whole.

Kernel-weighted covariances are built in two steps:

1. :func:`build_weight_matrix` — builds the (n, n) weight matrix ``W`` from
   coordinates or distances.  If ``kernel_kwargs`` contains ``support``, the
   kernel is exactly zero beyond that distance and ``W`` is built sparse from
   a parallel grid-based radius-neighbor search; otherwise ``W`` is dense
   (all pairs).
2. :func:`weighted_covariance` — computes ``(X - 1μᵀ)ᵀ W (X - 1μᵀ) / c``.

Dispatch pattern in :func:`weighted_covariance`:

* ``W=None``,   sparse X  →  :func:`_cov_sparse_standard`
* ``W=None``,   dense X   →  :func:`_cov_dense_standard`
* sparse ``W``            →  :func:`_cov_sparse_kernel`  (parallel, gene blocks)
* dense ``W``             →  :func:`_cov_dense_kernel`   (Numba)

Normalization ``c``: ``n - 1`` (``bias=False``) or ``n`` (``bias=True``), so
that ``W = I`` recovers the standard covariance.  For kernel-weighted
covariances, ``normalize_by_weights=True`` instead divides by
``S0 = Σ_ij W_ij`` (as in Moran's I), controlling for the total weight in ``W``.

Parallelism: functions taking ``n_jobs`` run their Numba kernels and BLAS
calls on that many threads (capped at the number of cores Numba sees).
"""

import itertools
import os
import warnings
from contextlib import contextmanager

import numba
import numpy as np
import scipy.sparse
from numba import njit, prange
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist as _pdist
from threadpoolctl import threadpool_limits

from .misc import squareform_from_condensed


# parallelism ───────────────────────────────────────────────────────────────

DEFAULT_N_JOBS = 16


@contextmanager
def _n_threads(n_jobs):
    """
    Run Numba kernels and BLAS calls on *n_jobs* threads.

    ``None`` or ``-1`` uses all cores.  Numba threads are capped at
    ``numba.config.NUMBA_NUM_THREADS`` (the number of cores by default).
    """
    n = os.cpu_count() if n_jobs in (None, -1) else int(n_jobs)
    if n < 1:
        raise ValueError(f"n_jobs must be a positive integer or -1, got {n_jobs}.")
    previous = numba.get_num_threads()
    numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    try:
        with threadpool_limits(limits=n):
            yield
    finally:
        numba.set_num_threads(previous)


# kernel functions ──────────────────────────────────────────────────────────

_BUILTIN_KERNELS = frozenset({'gaussian', 'inverse_distance', 'wendland'})
_KERNEL_CODES = {'gaussian': 0, 'inverse_distance': 1, 'wendland': 2}

# Above this many observations a dense (n, n) W is not built; a support is
# chosen automatically (Gaussian) or required (other kernels).
_MAX_DENSE_OBS = 100_000

# Automatic support for large datasets, as a multiple of the Gaussian bandwidth.
# The Wendland taper then shrinks the effective bandwidth by only ~9%.
_AUTO_SUPPORT_FACTOR = 10.0

# Refuse to build a sparse W with more stored entries than this (~12 GB).
_MAX_SPARSE_NNZ = 1_000_000_000

# Number of points subsampled to estimate the median distance (and the
# number of neighbors on the fallback search path) for large datasets.
_N_SUBSAMPLE = 2000

# Number of gene columns processed at once in the sparse-W covariance.
# Small blocks keep the neighbors' rows of each block in cache.
_GENE_BLOCK = 64


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


@njit(parallel=True)
def _kernel_inplace(values, code, param, support):
    """
    Overwrite distances with kernel weights in one parallel pass.

    ``code``: 0 Gaussian (``param`` = bandwidth), 1 inverse distance
    (``param`` = buffer), 2 Wendland (``support``), -1 taper only.  For
    codes 0, 1 and -1, ``support > 0`` multiplies by a Wendland taper.
    """
    inv_two_bw2 = 1.0 / (2.0 * param * param) if code == 0 else 0.0
    for i in prange(values.shape[0]):
        d = values[i]
        if code == 0:
            w = np.exp(-d * d * inv_two_bw2)
        elif code == 1:
            w = 1.0 / (d + param)
        elif code == 2:
            w = 1.0
        else:
            w = 1.0
        if support > 0.0:
            r = d / support
            if r < 1.0:
                t = 1.0 - r
                w *= t ** 4 * (4.0 * r + 1.0)
            else:
                w = 0.0
        values[i] = w


def _apply_kernel_inplace(values, kernel, kw):
    """
    Turn a float64 array of distances into built-in kernel weights in place
    (one parallel pass, taper included).  Returns the same array.
    """
    support = float(kw['support']) if 'support' in kw else 0.0

    if kernel not in _KERNEL_CODES:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Built-in options: {sorted(_BUILTIN_KERNELS)}. "
            "Or pass a callable with signature f(distances, **kwargs) -> weights."
        )
    if kernel == 'wendland' and support == 0.0:
        raise ValueError("kernel='wendland' requires a 'support' entry in kernel_kwargs.")

    if kernel == 'gaussian':
        param = float(kw['bandwidth']) if 'bandwidth' in kw else float(np.median(values))
    elif kernel == 'inverse_distance':
        param = float(kw.get('buffer', 1e-6))
    else:
        param = 0.0

    _kernel_inplace(values, _KERNEL_CODES[kernel], param, support)
    return values


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
    d = np.array(distances, dtype=np.float64, copy=True).ravel()

    if callable(kernel):
        call_kw = {k: v for k, v in kw.items() if k != 'support'}
        weights = np.asarray(kernel(d.copy(), **call_kw), dtype=np.float64)
        if 'support' in kw:
            _kernel_inplace(d, -1, 0.0, float(kw['support']))   # d -> taper
            weights = weights * d
        return weights

    return _apply_kernel_inplace(d, kernel, kw)


def _kernel_at_zero(kernel, kernel_kwargs=None):
    """Kernel value at zero distance, used as the diagonal of ``W``."""
    kw = kernel_kwargs or {}
    if kernel in ('gaussian', 'wendland'):
        return 1.0
    if kernel == 'inverse_distance':
        return 1.0 / kw.get('buffer', 1e-6)
    return float(apply_kernel(np.zeros(1), kernel, kw)[0])


def _warn_if_diagonal_dominates(diag_value, max_offdiag):
    """Warn when ``W`` is effectively a scaled identity matrix."""
    if diag_value > 1e3 * max_offdiag:
        warnings.warn(
            f"Kernel value at zero distance ({diag_value:.3g}) is more than 1000x the "
            f"largest off-diagonal weight ({max_offdiag:.3g}), so W is close to a scaled "
            "identity and the kernel has little effect.  For 'inverse_distance', "
            "set kernel_kwargs['buffer'] on the scale of the distances."
        )


# radius-neighbor search ────────────────────────────────────────────────────

# Grid search is used for up to 3-D coordinates when the grid has at most
# this many cells per observation (otherwise fall back to sklearn).
_MAX_GRID_CELLS_PER_OBS = 16


@njit(parallel=True)
def _grid_count(coords, cell_idx, n_cells, strides, order, cell_start, offsets, r2):
    """Number of points within sqrt(r2) of each point (including itself)."""
    n, dim = coords.shape
    counts = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for o in range(offsets.shape[0]):
            cell = 0
            inside = True
            for k in range(dim):
                q = cell_idx[i, k] + offsets[o, k]
                if q < 0 or q >= n_cells[k]:
                    inside = False
                    break
                cell += q * strides[k]
            if not inside:
                continue
            for t in range(cell_start[cell], cell_start[cell + 1]):
                j = order[t]
                s = 0.0
                for k in range(dim):
                    diff = coords[i, k] - coords[j, k]
                    s += diff * diff
                if s < r2:
                    c += 1
        counts[i] = c
    return counts


@njit(parallel=True)
def _grid_fill(coords, cell_idx, n_cells, strides, order, cell_start, offsets, r2,
               indptr, indices, distances):
    """Write neighbor indices and distances into preallocated CSR arrays."""
    n, dim = coords.shape
    for i in prange(n):
        pos = indptr[i]
        for o in range(offsets.shape[0]):
            cell = 0
            inside = True
            for k in range(dim):
                q = cell_idx[i, k] + offsets[o, k]
                if q < 0 or q >= n_cells[k]:
                    inside = False
                    break
                cell += q * strides[k]
            if not inside:
                continue
            for t in range(cell_start[cell], cell_start[cell + 1]):
                j = order[t]
                s = 0.0
                for k in range(dim):
                    diff = coords[i, k] - coords[j, k]
                    s += diff * diff
                if s < r2:
                    indices[pos] = j
                    distances[pos] = np.sqrt(s)
                    pos += 1


@njit(parallel=True)
def _max_offdiag(indptr, indices, data):
    """Largest off-diagonal entry of a CSR matrix."""
    n = indptr.shape[0] - 1
    row_max = np.zeros(n)
    for i in prange(n):
        m = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            if indices[t] != i and data[t] > m:
                m = data[t]
        row_max[i] = m
    return row_max.max() if n > 0 else 0.0


def _check_nnz(nnz, n_obs, support):
    if nnz > _MAX_SPARSE_NNZ:
        raise ValueError(
            f"support={support:.4g} gives about {nnz / n_obs:.0f} neighbors per "
            f"observation (~{nnz:.2g} stored weights, ~{nnz * 12 / 1e9:.0f} GB). "
            "Use a smaller kernel_kwargs['support'] (or a smaller 'bandwidth' for "
            "'gaussian', which sets the automatic support)."
        )


def _radius_neighbors_grid(coords, radius):
    """
    Pairs closer than *radius* (including each point with itself) as CSR
    arrays ``(indptr, indices, distances)``, via a parallel uniform-grid
    search.  Returns ``None`` if the grid would be too large.
    """
    n, dim = coords.shape
    lo = coords.min(axis=0)
    cell_idx = np.floor((coords - lo) / radius).astype(np.int64)
    n_cells = cell_idx.max(axis=0) + 1
    total = int(np.prod(n_cells.astype(np.float64)))
    if dim > 3 or total > _MAX_GRID_CELLS_PER_OBS * max(n, 1):
        return None

    strides = np.ones(dim, dtype=np.int64)
    for k in range(dim - 2, -1, -1):
        strides[k] = strides[k + 1] * n_cells[k + 1]
    cell_id = cell_idx @ strides
    order = np.argsort(cell_id, kind='stable')
    cell_start = np.searchsorted(cell_id[order], np.arange(total + 1)).astype(np.int64)
    offsets = np.array(list(itertools.product((-1, 0, 1), repeat=dim)), dtype=np.int64)
    r2 = radius * radius

    counts = _grid_count(coords, cell_idx, n_cells, strides, order, cell_start, offsets, r2)
    nnz = int(counts.sum())
    _check_nnz(nnz, n, radius)

    index_dtype = np.int32 if max(nnz, n) < np.iinfo(np.int32).max else np.int64
    indptr = np.zeros(n + 1, dtype=index_dtype)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(nnz, dtype=index_dtype)
    distances = np.empty(nnz, dtype=np.float64)
    _grid_fill(coords, cell_idx, n_cells, strides, order, cell_start, offsets, r2,
               indptr, indices, distances)
    return indptr, indices, distances


def _subsample(coords, rng=None):
    """Random subsample of at most ``_N_SUBSAMPLE`` rows."""
    n = coords.shape[0]
    if n <= _N_SUBSAMPLE:
        return coords
    rng = np.random.default_rng(0) if rng is None else rng
    return coords[rng.choice(n, _N_SUBSAMPLE, replace=False)]


def _radius_neighbors_fallback(coords, radius, n_jobs):
    """
    sklearn radius search (used for >3-D coordinates or very sparse grids).
    Unlike the grid search, the diagonal is not included.
    """
    from sklearn.neighbors import radius_neighbors_graph

    counts = cKDTree(coords).query_ball_point(_subsample(coords), r=radius, return_length=True)
    _check_nnz(float(np.mean(counts)) * coords.shape[0], coords.shape[0], radius)
    G = radius_neighbors_graph(coords, radius=radius, mode='distance', n_jobs=n_jobs).tocsr()
    return G.indptr, G.indices, np.ascontiguousarray(G.data, dtype=np.float64)


# weight matrix ─────────────────────────────────────────────────────────────

def build_weight_matrix(
    n_obs,
    *,
    kernel,
    coordinates=None,
    distances=None,
    kernel_kwargs=None,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Build the symmetric (n, n) kernel weight matrix ``W``.

    Distances are Euclidean.  The diagonal is the kernel value at zero
    distance, so ``W`` is positive semi-definite whenever the kernel is.

    **Sparse vs dense.**  If ``kernel_kwargs`` contains ``support`` (required
    for ``'wendland'``), weights are exactly zero beyond that distance, so
    ``W`` is built sparse from the pairs closer than ``support`` (parallel
    grid search for up to 3-D coordinates).  For other kernels, ``support``
    tapers the kernel by a Wendland kernel (``W = K ∘ K_Wendland``), which
    keeps ``W`` PSD; a hard cutoff alone would not.  Without ``support``,
    ``W`` is dense.

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
    n_jobs : int, default 16
        Number of threads for the neighbor search and kernel evaluation.
        ``-1`` uses all cores.

    Returns
    -------
    W : (n, n) float64 ndarray or scipy.sparse CSR matrix
    """
    if coordinates is None and distances is None:
        raise ValueError("Supply `coordinates` or `distances` when using a kernel.")

    with _n_threads(n_jobs):
        return _build_weight_matrix(n_obs, kernel, coordinates, distances,
                                    dict(kernel_kwargs or {}), n_jobs)


def _dense_from_condensed(weights, diag_value):
    W = squareform_from_condensed(np.ascontiguousarray(weights, dtype=np.float64))
    _warn_if_diagonal_dominates(diag_value, weights.max() if len(weights) else 0.0)
    np.fill_diagonal(W, diag_value)
    return W


def _build_weight_matrix(n_obs, kernel, coordinates, distances, kw, n_jobs):
    coords = None if coordinates is None else np.ascontiguousarray(coordinates, dtype=np.float64)

    if coords is not None and coords.shape[1] > 3 and (kernel == 'wendland' or 'support' in kw):
        warnings.warn(
            f"Coordinates have {coords.shape[1]} dimensions; the Wendland kernel "
            "(standalone or as a taper) is only guaranteed PSD in up to 3 dimensions."
        )

    # dense path from precomputed distances
    if distances is not None:
        weights = apply_kernel(distances, kernel, kernel_kwargs=kw)
        return _dense_from_condensed(weights, _kernel_at_zero(kernel, kw))

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

    # dense path: all pairs
    if 'support' not in kw:
        weights = apply_kernel(_pdist(coords), kernel, kernel_kwargs=kw)
        return _dense_from_condensed(weights, _kernel_at_zero(kernel, kw))

    # sparse path: only pairs closer than the support (each point with itself
    # included, so the diagonal is the kernel value at zero distance)
    support = float(kw['support'])
    found = _radius_neighbors_grid(coords, support)
    self_included = found is not None
    if not self_included:
        found = _radius_neighbors_fallback(coords, support, n_jobs)
    indptr, indices, values = found

    if callable(kernel):
        values = apply_kernel(values, kernel, kernel_kwargs=kw)
    else:
        _apply_kernel_inplace(values, kernel, kw)

    W = scipy.sparse.csr_matrix((values, indices, indptr), shape=(n_obs, n_obs))
    diag_value = _kernel_at_zero(kernel, kw)
    _warn_if_diagonal_dominates(diag_value, _max_offdiag(indptr, indices, values))
    if not self_included:
        W = (W + scipy.sparse.diags(np.full(n_obs, diag_value), format='csr')).tocsr()
    return W


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


@njit(parallel=True)
def _csr_matmat_permuted(indptr, indices, data, Bp, perm, inv):
    """
    ``(W @ B)[perm]`` for CSR ``W`` given ``Bp = B[perm]`` (C-contiguous),
    parallel over rows.  ``inv`` is the inverse permutation.  Visiting rows
    in a bandwidth-reducing order keeps neighbors' rows of ``Bp`` in cache.
    """
    n = indptr.shape[0] - 1
    b = Bp.shape[1]
    out = np.zeros((n, b))
    for ii in prange(n):
        i = perm[ii]
        for t in range(indptr[i], indptr[i + 1]):
            jj = inv[indices[t]]
            v = data[t]
            for k in range(b):
                out[ii, k] += v * Bp[jj, k]
    return out


@njit(parallel=True)
def _csc_t_matmat(indptr, indices, data, B):
    """``Xᵀ @ B`` for CSC ``X`` and dense C-contiguous ``B``, parallel over columns."""
    p = indptr.shape[0] - 1
    b = B.shape[1]
    out = np.zeros((p, b))
    for j in prange(p):
        for t in range(indptr[j], indptr[j + 1]):
            i = indices[t]
            v = data[t]
            for k in range(b):
                out[j, k] += v * B[i, k]
    return out


def _cov_sparse_kernel(X, W, denom):
    """
    Kernel-weighted covariance for sparse W (X dense or sparse).

    ``XᵀWX`` is computed in blocks of ``_GENE_BLOCK`` genes: each block of X
    is densified, multiplied by W with a parallel CSR kernel, and multiplied
    by Xᵀ (BLAS for dense X, a parallel CSC kernel for sparse X).  Rows are
    visited in reverse Cuthill-McKee order, which puts spatial neighbors
    close together in memory; since ``XᵀWX = (PX)ᵀ (PWPᵀ) (PX)`` for any
    permutation ``P``, the result is unchanged.  Memory is O(n · block)
    beyond X and W (plus one row-permuted copy of X when X is sparse).

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
    W = W.tocsr()
    data = np.ascontiguousarray(W.data, dtype=np.float64)
    p = X.shape[1]

    mu = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()
    w = np.asarray(W.sum(axis=1), dtype=np.float64).ravel()   # (n,)
    Xw = np.asarray(X.T @ w, dtype=np.float64).ravel()          # (p,)
    S_W = w.sum()

    perm = reverse_cuthill_mckee(W, symmetric_mode=True).astype(np.int64)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.shape[0])

    X_is_sparse = scipy.sparse.issparse(X)
    if X_is_sparse:
        Xc = X.tocsr()[perm].tocsc()       # rows in RCM order, columns sliceable
        Xc_data = np.ascontiguousarray(Xc.data, dtype=np.float64)

    XtWX = np.empty((p, p))
    for j0 in range(0, p, _GENE_BLOCK):
        j1 = min(j0 + _GENE_BLOCK, p)
        if X_is_sparse:
            Bp = np.ascontiguousarray(Xc[:, j0:j1].toarray(), dtype=np.float64)
        else:
            Bp = np.ascontiguousarray(X[perm, j0:j1], dtype=np.float64)
        WBp = _csr_matmat_permuted(W.indptr, W.indices, data, Bp, perm, inv)   # (W X_b)[perm]
        if X_is_sparse:
            XtWX[:, j0:j1] = _csc_t_matmat(Xc.indptr, Xc.indices, Xc_data, WBp)
        else:
            WXb = np.empty_like(WBp)
            WXb[perm] = WBp
            XtWX[:, j0:j1] = X.T @ WXb

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


def weighted_covariance(X, W=None, *, bias=False, normalize_by_weights=False,
                        n_jobs=DEFAULT_N_JOBS):
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
    n_jobs : int, default 16
        Number of threads for the matrix products.  ``-1`` uses all cores.

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    with _n_threads(n_jobs):
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
            return _cov_sparse_kernel(X, W, denom)

        # W is dense; densify X if needed and use the Numba path
        if X_is_sparse:
            X = X.toarray()
        return _cov_dense_kernel(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(W, dtype=np.float64),
            denom,
        )


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
    n_jobs=DEFAULT_N_JOBS,
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
    n_jobs : int, default 16
        Number of threads for the neighbor search, kernel evaluation, and
        matrix products.  ``-1`` uses all cores.

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
            n_jobs=n_jobs,
        )

    cov = weighted_covariance(X, W, bias=bias, normalize_by_weights=normalize_by_weights,
                              n_jobs=n_jobs)
    return (cov, W) if return_weights else cov
