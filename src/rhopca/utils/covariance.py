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

:func:`grouped_weighted_variance` computes only the diagonal of that
covariance, separately for each group of rows (W restricted to within-group
pairs), in a single pass over W — used for per-group rho values.

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
import scipy.fft
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

# Refuse to build a sparse W with more stored entries than this (~8 GB).
_MAX_SPARSE_NNZ = 1_000_000_000

# Number of points subsampled to estimate the default bandwidth (and the
# number of neighbors on the fallback search path) for large datasets.
_N_SUBSAMPLE = 2000

# Default Gaussian bandwidth: median distance to the k-th nearest neighbor,
# k = min(_DEFAULT_NEIGHBORS, n_obs // _NEIGHBORS_PER_OBS), at least 1.
_DEFAULT_NEIGHBORS = 100
_NEIGHBORS_PER_OBS = 200

# Number of gene columns processed at once in the sparse-W covariance.
# Small blocks keep the neighbors' rows of each block in cache.
_GENE_BLOCK = 64

# Largest distance from the fitted lattice, as a fraction of the spacing, for
# coordinates to count as a regular grid.
_GRID_TOL = 0.05

# Largest number of lattice sites per observation for the FFT path.
_MAX_LATTICE_SITES_PER_OBS = 16

# Number of columns transformed at once on the FFT path.
_FFT_BLOCK = 16

# Gaussian weights are below 1e-13 beyond this many bandwidths.
_GAUSSIAN_REACH = 8.0


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


class _TooManyNeighbors(ValueError):
    def __init__(self, nnz, n_obs, support):
        self.nnz, self.n_obs, self.support = nnz, n_obs, support
        super().__init__(
            f"support={support:.4g} gives about {nnz / n_obs:,.0f} neighbors per "
            f"observation (~{nnz:.2g} stored weights, ~{nnz * 8 / 1e9:.0f} GB). "
            "Use a smaller kernel_kwargs['support']."
        )


def _check_nnz(nnz, n_obs, support):
    if nnz > _MAX_SPARSE_NNZ:
        raise _TooManyNeighbors(nnz, n_obs, support)


def _too_many_neighbors_message(err, kernel, kw, default_k, auto_support):
    """Error message for a sparse ``W`` over the size limit."""
    size = (f"gives about {err.nnz / err.n_obs:,.0f} neighbors per observation "
            f"(~{err.nnz:.2g} stored weights, ~{err.nnz * 8 / 1e9:.0f} GB).")
    if kernel != 'gaussian':
        return f"support = {err.support:.4g} {size} Pass a smaller kernel_kwargs['support']."
    if default_k is None:
        bandwidth = f"bandwidth = {kw['bandwidth']:.4g}"
    else:
        bandwidth = (f"Default bandwidth = {kw['bandwidth']:.4g} (median distance to "
                     f"the {default_k}th nearest neighbor)")
    if auto_support:
        return (f"{bandwidth} with automatic support = {_AUTO_SUPPORT_FACTOR:g} x bandwidth "
                f"= {err.support:.4g} {size} Lower kernel_kwargs['bandwidth'] or pass "
                f"kernel_kwargs['support'] below {err.support:.4g}.")
    return (f"{bandwidth} with support = {err.support:.4g} {size} "
            "Pass a smaller kernel_kwargs['support'].")


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
    distances = np.empty(nnz, dtype=np.float32)
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
    return G.indptr, G.indices, np.ascontiguousarray(G.data, dtype=np.float32)


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
    ``ValueError`` is raised.

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
    W : (n, n) float64 ndarray or float32 scipy.sparse CSR matrix
    """
    if coordinates is None and distances is None:
        raise ValueError("Supply `coordinates` or `distances` when using a kernel.")

    with _n_threads(n_jobs):
        return _build_weight_matrix(n_obs, kernel, coordinates, distances,
                                    dict(kernel_kwargs or {}), n_jobs)


def _default_neighbors(n_obs):
    return max(1, min(_DEFAULT_NEIGHBORS, n_obs // _NEIGHBORS_PER_OBS))


def _resolve_bandwidth(kernel, kw, coords=None, distances=None, n_jobs=DEFAULT_N_JOBS):
    """
    Fill in the default Gaussian bandwidth: the median distance to the k-th
    nearest neighbor (see :func:`_default_neighbors`), over a subsample of
    points.  Returns k if the default was used, else ``None``.
    """
    if kernel != 'gaussian' or 'bandwidth' in kw:
        return None
    if distances is not None:
        D = squareform_from_condensed(np.ascontiguousarray(distances, dtype=np.float64))
        k = _default_neighbors(D.shape[0])
        rows = _subsample(np.arange(D.shape[0])[:, None]).ravel()
        kth = np.partition(D[rows], k, axis=1)[:, k]
    else:
        k = _default_neighbors(coords.shape[0])
        workers = -1 if n_jobs in (None, -1) else int(n_jobs)
        kth = cKDTree(coords).query(_subsample(coords), k=k + 1, workers=workers)[0][:, k]
    bandwidth = float(np.median(kth))
    if not bandwidth > 0:
        raise ValueError(
            "The default bandwidth is 0 (many duplicate coordinates); pass "
            "kernel_kwargs['bandwidth']."
        )
    kw['bandwidth'] = bandwidth
    return k


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
        _resolve_bandwidth(kernel, kw, distances=distances)
        weights = apply_kernel(distances, kernel, kernel_kwargs=kw)
        return _dense_from_condensed(weights, _kernel_at_zero(kernel, kw))

    large = n_obs >= _MAX_DENSE_OBS
    default_k = _resolve_bandwidth(kernel, kw, coords, n_jobs=n_jobs)

    auto_support = 'support' not in kw and large
    if auto_support:
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
    try:
        found = _radius_neighbors_grid(coords, support)
        self_included = found is not None
        if not self_included:
            found = _radius_neighbors_fallback(coords, support, n_jobs)
    except _TooManyNeighbors as err:
        raise ValueError(
            _too_many_neighbors_message(err, kernel, kw, default_k, auto_support)
        ) from None
    indptr, indices, values = found

    if callable(kernel):
        values = apply_kernel(values, kernel, kernel_kwargs=kw)
    else:
        _apply_kernel_inplace(values, kernel, kw)

    values = values.astype(np.float32, copy=False)
    W = scipy.sparse.csr_matrix((values, indices, indptr), shape=(n_obs, n_obs))
    diag_value = _kernel_at_zero(kernel, kw)
    _warn_if_diagonal_dominates(diag_value, _max_offdiag(indptr, indices, values))
    if not self_included:
        diagonal = np.full(n_obs, diag_value, dtype=np.float32)
        W = (W + scipy.sparse.diags(diagonal, format='csr')).tocsr()
    return W


# regular-grid weights ──────────────────────────────────────────────────────

def _lattice_basis(coords):
    """Estimate two lattice vectors and the spacing from neighbor offsets."""
    query = _subsample(coords)
    k = min(9, coords.shape[0])
    _, nbrs = cKDTree(coords).query(query, k=k)
    offsets = (coords[nbrs[:, 1:]] - query[:, None, :]).reshape(-1, 2)
    norms = np.linalg.norm(offsets, axis=1)
    if k < 3 or norms.min() == 0:
        return None

    spacing = float(np.median(norms.reshape(query.shape[0], -1).min(axis=1)))
    first = offsets[norms <= 1.25 * spacing]
    a = np.median(first[np.linalg.norm(first - first[0], axis=1) < 0.5 * spacing], axis=0)

    cross = np.abs(a[0] * offsets[:, 1] - a[1] * offsets[:, 0])
    apart = offsets[cross > 0.5 * np.linalg.norm(a) * norms]
    if apart.shape[0] == 0:
        return None
    apart_norms = np.linalg.norm(apart, axis=1)
    nearest = apart[apart_norms <= 1.25 * np.median(apart_norms[apart_norms <= 1.25 * apart_norms.min()])]
    b = np.median(nearest[np.linalg.norm(nearest - nearest[0], axis=1) < 0.5 * spacing], axis=0)
    return np.column_stack([a, b]), spacing


def _detect_lattice(coords):
    """
    Fit a 2-D lattice to *coords*.

    Returns ``(idx, basis)``, with ``idx`` the (n, 2) non-negative lattice
    sites and ``basis`` the (2, 2) lattice vectors, or ``None`` if the
    coordinates are not approximately a regular grid.
    """
    if coords.shape[1] != 2:
        return None
    found = _lattice_basis(coords)
    if found is None:
        return None
    basis, spacing = found

    # The basis is refined outward from the center so that a small error in
    # the first estimate cannot accumulate into wrong sites far from it.
    centered = coords - coords[np.argmin(((coords - np.median(coords, axis=0)) ** 2).sum(axis=1))]
    dist = np.linalg.norm(centered, axis=1)
    radius = 20.0 * spacing
    while True:
        idx = np.rint(np.linalg.solve(basis, centered.T).T)
        inside = dist <= radius
        design = np.column_stack([idx[inside], np.ones(inside.sum())])
        if design.shape[0] < 3 or np.linalg.matrix_rank(design) < 3:
            return None
        coef = np.linalg.lstsq(design, centered[inside], rcond=None)[0]
        basis = coef[:2].T
        if inside.all():
            break
        radius *= 2.0

    idx = np.rint(np.linalg.solve(basis, (centered - coef[2]).T).T).astype(np.int64)
    residual = centered - idx @ basis.T - coef[2]
    if np.sqrt((residual ** 2).sum(axis=1).max()) > _GRID_TOL * spacing:
        return None

    idx -= idx.min(axis=0)
    shape = idx.max(axis=0) + 1
    if np.unique(idx[:, 0] * shape[1] + idx[:, 1]).size != coords.shape[0]:
        return None
    if shape.prod() > _MAX_LATTICE_SITES_PER_OBS * coords.shape[0]:
        return None
    return idx, basis


class _GridKernel:
    """
    Kernel weights on a lattice, applied by FFT without storing ``W``.

    Parameters
    ----------
    idx : (n, 2) int array
        Lattice sites from :func:`_detect_lattice`.
    basis : (2, 2) float array
        Lattice vectors.
    kernel : str or callable
        See :func:`apply_kernel`.
    kw : dict
        Kernel parameters, with the bandwidth resolved.
    n_jobs : int
        Number of FFT workers.
    """

    def __init__(self, idx, basis, kernel, kw, n_jobs):
        self.rows, self.cols = idx[:, 0], idx[:, 1]
        self.shape = (idx.shape[0], idx.shape[0])
        self.workers = os.cpu_count() if n_jobs in (None, -1) else int(n_jobs)

        sites = idx.max(axis=0) + 1
        if 'support' in kw:
            reach = float(kw['support'])
        elif kernel == 'gaussian':
            reach = _GAUSSIAN_REACH * float(kw['bandwidth'])
        else:
            reach = np.inf
        per_axis = reach * np.linalg.norm(np.linalg.inv(basis), axis=1)
        pad = np.minimum(np.ceil(np.nan_to_num(per_axis, posinf=sites.max())), sites - 1).astype(np.int64)
        self.grid = tuple(scipy.fft.next_fast_len(int(s + p), real=True)
                          for s, p in zip(sites, pad))

        # Signed lattice offsets in FFT order; those beyond pad stay zero so
        # that the circular convolution has no wrap-around.
        steps = [np.arange(m) for m in self.grid]
        for step, m in zip(steps, self.grid):
            step[step > m // 2] -= m
        d_row, d_col = np.meshgrid(*steps, indexing='ij')
        offsets = np.stack([d_row, d_col], axis=-1) @ basis.T
        weights = apply_kernel(np.linalg.norm(offsets, axis=-1), kernel, kw).reshape(self.grid)
        weights[(np.abs(d_row) > pad[0]) | (np.abs(d_col) > pad[1])] = 0.0

        origin = weights[0, 0]
        weights[0, 0] = 0.0
        _warn_if_diagonal_dominates(origin, weights.max())
        weights[0, 0] = origin
        self.spectrum = scipy.fft.rfft2(weights, workers=self.workers)
        self._total = None

    def matmat(self, Z, subset=None):
        """
        ``W @ Z``, or ``W[subset][:, subset] @ Z`` for ``subset`` row indices.
        """
        rows = self.rows if subset is None else self.rows[subset]
        cols = self.cols if subset is None else self.cols[subset]
        out = np.empty_like(Z)
        for j0 in range(0, Z.shape[1], _FFT_BLOCK):
            block = Z[:, j0:j0 + _FFT_BLOCK]
            grid = np.zeros((block.shape[1],) + self.grid)
            grid[:, rows, cols] = block.T
            freq = scipy.fft.rfft2(grid, axes=(1, 2), workers=self.workers)
            freq *= self.spectrum
            grid = scipy.fft.irfft2(freq, s=self.grid, axes=(1, 2), workers=self.workers)
            out[:, j0:j0 + _FFT_BLOCK] = grid[:, rows, cols].T
        return out

    def row_sums(self, subset=None):
        """Row sums of ``W``, or of ``W[subset][:, subset]``."""
        size = self.shape[0] if subset is None else len(subset)
        return self.matmat(np.ones((size, 1)), subset).ravel()

    def sum(self):
        """Total weight ``S0``."""
        if self._total is None:
            self._total = float(self.row_sums().sum())
        return self._total


def _weight_operator(n_obs, *, kernel, coordinates, kernel_kwargs=None, n_jobs=DEFAULT_N_JOBS):
    """
    Weights for :func:`weighted_covariance`: a :class:`_GridKernel` for at
    least ``_MAX_DENSE_OBS`` observations on a regular grid, otherwise the
    matrix from :func:`build_weight_matrix`.
    """
    if n_obs >= _MAX_DENSE_OBS:
        coords = np.ascontiguousarray(coordinates, dtype=np.float64)
        lattice = _detect_lattice(coords)
        if lattice is not None:
            kw = dict(kernel_kwargs or {})
            _resolve_bandwidth(kernel, kw, coords, n_jobs=n_jobs)
            with _n_threads(n_jobs):
                return _GridKernel(*lattice, kernel, kw, n_jobs)
    return build_weight_matrix(n_obs, kernel=kernel, coordinates=coordinates,
                               kernel_kwargs=kernel_kwargs, n_jobs=n_jobs)


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
    X = X.astype(np.float64, copy=False)
    mu = _column_means(X)
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


def rcm_order(W):
    """Reverse Cuthill-McKee row order of a symmetric sparse ``W`` (int64)."""
    return reverse_cuthill_mckee(W.tocsr(), symmetric_mode=True).astype(np.int64)


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
    data = _sparse_values(W)
    p = X.shape[1]

    mu = _column_means(X)
    w = W @ np.ones(W.shape[0])                                 # (n,)
    Xw = np.asarray(X.T @ w, dtype=np.float64).ravel()          # (p,)
    S_W = w.sum()

    perm = rcm_order(W)
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


def _cov_grid_kernel(X, W, denom):
    """
    Kernel-weighted covariance for a :class:`_GridKernel` (X dense or sparse).

    Parameters
    ----------
    X : (n, p) sparse matrix or float64 ndarray
    W : _GridKernel
    denom : float
        Normalization constant (n-1, n, or S0).

    Returns
    -------
    cov : (p, p) float64 ndarray
    """
    X_is_sparse = scipy.sparse.issparse(X)
    if X_is_sparse:
        X = X.tocsc().astype(np.float64, copy=False)
    mu = _column_means(X)
    p = X.shape[1]

    cov = np.empty((p, p))
    for j0 in range(0, p, _FFT_BLOCK):
        j1 = min(j0 + _FFT_BLOCK, p)
        block = X[:, j0:j1].toarray() if X_is_sparse else X[:, j0:j1]
        WB = W.matmat(np.asarray(block, dtype=np.float64) - mu[j0:j1])
        cov[:, j0:j1] = _xt_times(X, WB) - np.outer(mu, WB.sum(axis=0))
    return (cov + cov.T) / (2.0 * denom)


def _column_means(X):
    """Column means of dense or sparse ``X``, summed in float64."""
    if scipy.sparse.issparse(X):
        return (X.T @ np.ones(X.shape[0])) / X.shape[0]
    return X.mean(axis=0, dtype=np.float64)


def _xt_times(X, B):
    """``Xᵀ @ B`` for dense ``B``; ``X`` is float64 CSC when sparse."""
    if scipy.sparse.issparse(X):
        return _csc_t_matmat(X.indptr, X.indices, X.data, np.ascontiguousarray(B))
    return X.T @ B


def _sparse_values(W):
    """Stored values of sparse ``W`` as floats, without copying them."""
    if W.data.dtype in (np.float32, np.float64):
        return W.data
    return W.data.astype(np.float64)


def _denominator(W, n, bias, normalize_by_weights):
    if normalize_by_weights:
        if scipy.sparse.issparse(W):
            W = W @ np.ones(W.shape[0])     # sum in float64
        denom = float(W.sum())
        if denom <= 0:
            raise ValueError(f"Sum of weights S0 = {denom:.3g} must be positive.")
        return denom
    return float(n if bias else n - 1)


def weighted_covariance(X, W=None, *, bias=False, normalize_by_weights=False,
                        n_jobs=DEFAULT_N_JOBS):
    """
    Compute ``(X - 1μᵀ)ᵀ W (X - 1μᵀ) / c`` for a given weight matrix.

    Parameters
    ----------
    X : (n, p) array-like — dense ndarray or scipy.sparse matrix
        Raw or scaled data matrix.  Centering is handled internally.
    W : (n, n) ndarray, scipy.sparse matrix, _GridKernel, or None
        Symmetric weights.  ``None`` is equivalent to the identity and
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

        denom = _denominator(W, X.shape[0], bias, normalize_by_weights)

        if isinstance(W, _GridKernel):
            if not X_is_sparse:
                X = np.asarray(X, dtype=np.float64)
            return _cov_grid_kernel(X, W, denom)

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


@njit(parallel=True)
def _grouped_quadform_rows(indptr, indices, data, Pp, labels_p, perm, inv, qp, sp):
    """
    Per-row terms of ``diag(P_gᵀ W_gg P_g)`` for every group at once.

    For row i in group g, ``q[i] = P[i] ∘ Σ_l W_il P[l]`` and
    ``s[i] = Σ_l W_il``, summing only over neighbors l in the same group.
    Rows are visited in the order ``perm`` (as in :func:`_csr_matmat_permuted`);
    inputs ``Pp``, ``labels_p`` and outputs ``qp``, ``sp`` are in that order.
    """
    n, k = Pp.shape
    for ii in prange(n):
        i = perm[ii]
        g = labels_p[ii]
        acc = np.zeros(k)
        wi = 0.0
        for t in range(indptr[i], indptr[i + 1]):
            jj = inv[indices[t]]
            if labels_p[jj] == g:
                w = data[t]
                wi += w
                for c in range(k):
                    acc[c] += w * Pp[jj, c]
        for c in range(k):
            qp[ii, c] = Pp[ii, c] * acc[c]
        sp[ii] = wi


def _subspace_quadratic(X, U, W, *, bias=False, normalize_by_weights=False, order=None,
                        n_jobs=DEFAULT_N_JOBS):
    """
    Compute ``Uᵀ (X - 1μᵀ)ᵀ W (X - 1μᵀ) U / c`` for a gene basis ``U``.

    Parameters
    ----------
    X : (n, p) array-like — dense ndarray or scipy.sparse matrix
    U : (p, k) float64 ndarray
    W : (n, n) ndarray, scipy.sparse matrix, or _GridKernel
    bias, normalize_by_weights : bool
        Normalization ``c``, as in :func:`weighted_covariance`.
    order : (n,) int array, optional
        Row visiting order for sparse *W*, from :func:`rcm_order`.
    n_jobs : int, default 16
        Number of threads.  ``-1`` uses all cores.

    Returns
    -------
    K : (k, k) float64 ndarray
    """
    with _n_threads(n_jobs):
        n, k = X.shape[0], U.shape[1]
        denom = _denominator(W, n, bias, normalize_by_weights)
        mu = _column_means(X)
        offset = mu @ U
        if scipy.sparse.issparse(X):
            X_t = X.tocsc().astype(np.float64, copy=False)
        else:
            X_t = np.asarray(X, dtype=np.float64)

        sparse_W = scipy.sparse.issparse(W)
        if sparse_W:
            W = W.tocsr()
            data = _sparse_values(W)
            perm = rcm_order(W) if order is None else np.asarray(order, dtype=np.int64)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(n)

        K = np.empty((k, k))
        for j0 in range(0, k, _GENE_BLOCK):
            j1 = min(j0 + _GENE_BLOCK, k)
            Z = np.asarray(X @ U[:, j0:j1]) - offset[j0:j1]
            if sparse_W:
                WZ = np.empty_like(Z)
                WZ[perm] = _csr_matmat_permuted(W.indptr, W.indices, data,
                                                np.ascontiguousarray(Z[perm]), perm, inv)
            elif isinstance(W, _GridKernel):
                WZ = W.matmat(Z)
            else:
                WZ = W @ Z
            K[:, j0:j1] = U.T @ (_xt_times(X_t, WZ) - np.outer(mu, WZ.sum(axis=0)))
        return (K + K.T) / (2.0 * denom)


def grouped_weighted_variance(P, W=None, labels=None, *, bias=False,
                              normalize_by_weights=False, order=None,
                              n_jobs=DEFAULT_N_JOBS):
    """
    Per-group, per-column kernel-weighted variances in a single pass over W.

    Row g of the result equals
    ``diag(weighted_covariance(P[labels == g], W[g][:, g], ...))``: W is
    restricted to pairs of cells in the same group, and each group is
    centered by its own mean.  Unlike calling :func:`weighted_covariance` per
    group, W is never sliced or copied, and only the diagonal is computed.

    Parameters
    ----------
    P : (n, k) array-like
        Data (e.g. projections onto components); dense.
    W : (n, n) ndarray, scipy.sparse matrix, _GridKernel, or None
        Symmetric weights.  ``None`` gives plain per-group variances.
    labels : (n,) int array, optional
        Group codes ``0 .. G-1``.  ``None`` puts all rows in one group.
    bias, normalize_by_weights : bool
        Normalization, as in :func:`weighted_covariance` (per group:
        ``n_g - 1``, ``n_g``, or ``S0_g = Σ_{i,l∈g} W_il``).
    order : (n,) int array, optional
        Row visiting order for sparse *W* (from :func:`rcm_order`).  Computed
        if not given; pass it to reuse across calls with the same *W*.
    n_jobs : int, default 16
        Number of threads.  ``-1`` uses all cores.

    Returns
    -------
    var : (G, k) float64 ndarray
        NaN where the normalization is not positive (e.g. a single-cell group
        with ``bias=False``).
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    n, k = P.shape
    labels = (np.zeros(n, dtype=np.int64) if labels is None
              else np.ascontiguousarray(labels, dtype=np.int64))
    n_groups = int(labels.max()) + 1 if n else 0
    counts = np.bincount(labels, minlength=n_groups).astype(np.float64)

    # (G, n) indicator: group sums of any per-row quantity are M @ values
    M = scipy.sparse.csr_matrix(
        (np.ones(n), (labels, np.arange(n))), shape=(n_groups, n)
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.asarray(M @ P) / counts[:, None]
    Pc = P - means[labels]   # each group centered by its own mean

    with _n_threads(n_jobs):
        if W is None:
            q, s = Pc * Pc, None
        elif isinstance(W, _GridKernel):
            q, s = np.empty((n, k)), np.empty(n)
            for g in range(n_groups):
                idx = np.flatnonzero(labels == g)
                q[idx] = W.matmat(Pc[idx], idx) * Pc[idx]
                s[idx] = W.row_sums(idx)
        elif scipy.sparse.issparse(W):
            W = W.tocsr()
            perm = rcm_order(W) if order is None else np.asarray(order, dtype=np.int64)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(n)
            qp, sp_ = np.empty((n, k)), np.empty(n)
            _grouped_quadform_rows(
                W.indptr, W.indices, _sparse_values(W),
                np.ascontiguousarray(Pc[perm]), labels[perm], perm, inv, qp, sp_,
            )
            q, s = np.empty((n, k)), np.empty(n)
            q[perm], s[perm] = qp, sp_
        else:
            W = np.asarray(W, dtype=np.float64)
            q, s = np.empty((n, k)), np.empty(n)
            for g in range(n_groups):
                idx = np.flatnonzero(labels == g)
                Wg = W[np.ix_(idx, idx)]
                q[idx] = (Wg @ Pc[idx]) * Pc[idx]
                s[idx] = Wg.sum(axis=1)

    num = np.asarray(M @ q)
    if W is not None and normalize_by_weights:
        denom = np.asarray(M @ s).ravel()
    else:
        denom = counts if bias else counts - 1
    out = np.full((n_groups, k), np.nan)
    ok = denom > 0
    out[ok] = num[ok] / denom[ok, None]
    return out


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
