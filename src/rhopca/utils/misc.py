"""
General utilities shared across rhoPCA classes.
"""

import numpy as np
import warnings
from scipy.linalg import eigh, eig
from numba import njit, prange
from sklearn.preprocessing import StandardScaler


def standardize_array(X):
    """
    Scale *X* to unit variance using ``sklearn.preprocessing.StandardScaler``
    with ``with_mean=False, with_std=True``.

    Centering is omitted here since it is handled implicitly inside
    ``compute_covariance`` via the algebraic expansion of the covariance
    formula.  This step preserves sparsity, and sparse matrices are passed 
    directly to ``StandardScaler`` without densification.

    Parameters
    ----------
    X : (n, p) array-like, dense ndarray, or sparse matrix

    Returns
    -------
    scaled : (n, p) array or sparse matrix
        Same type as input.
    """
    return StandardScaler(with_mean=False, with_std=True).fit_transform(X)


def generalized_eigen(Sigma_t, Sigma_b, *, method='schur', mu=None, n_components=None):
    """
    Generalized eigendecomposition of *Sigma_t* w.r.t. *Sigma_b*.

    Returns ``(eigvals, eigvecs)`` filtered to positive, finite values, sorted
    in descending order, and optionally truncated to *n_components*.

    Parameters
    ----------
    Sigma_t : (p, p) ndarray
        Target covariance matrix.
    Sigma_b : (p, p) ndarray
        Background covariance matrix.
    method : {'schur', 'tikhonov'}, default ``'schur'``
        Fallback solver used when *Sigma_b* is not positive definite.
        ``'schur'``    — generalized Schur decomposition (``scipy.linalg.eig``).
        ``'tikhonov'`` — adds a ridge ``mu * I`` to *Sigma_b*.
    mu : float, optional
        Ridge parameter for ``'tikhonov'``.  Defaults to
        ``1e-6 · trace(Sigma_b) / p``.
    n_components : int, optional
        Number of components to retain.  ``None`` keeps all valid components.

    Returns
    -------
    eigvals : (k,) float64 ndarray
    eigvecs : (p, k) float64 ndarray
    """
    try:
        eigvals, eigvecs = eigh(Sigma_t, Sigma_b)

    except np.linalg.LinAlgError:
        warnings.warn(
            "Background covariance is not positive definite; "
            f"falling back to method='{method}'."
        )
        if method == 'schur':
            eigvals, eigvecs = eig(Sigma_t, Sigma_b)
        elif method == 'tikhonov':
            if mu is None:
                mu = 1e-6 * np.trace(Sigma_b) / Sigma_b.shape[0]
            eigvals, eigvecs = eigh(Sigma_t, Sigma_b + mu * np.eye(Sigma_b.shape[0]))
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'schur' or 'tikhonov'.")

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    valid = (eigvals > 0) & np.isfinite(eigvals)
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if n_components is not None:
        eigvals = eigvals[:n_components]
        eigvecs = eigvecs[:, :n_components]

    return eigvals, eigvecs


@njit(parallel=True)
def squareform_from_condensed(condensed):
    """
    Convert a condensed distance vector to a symmetric square matrix.

    Parameters
    ----------
    condensed : 1-D float64 array
        Output of ``scipy.spatial.distance.pdist``.

    Returns
    -------
    W : (n, n) float64 ndarray
    """
    L = len(condensed)
    n = int((1 + np.sqrt(1 + 8 * L)) / 2)
    W = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        row_start = i * (n - 1) - i * (i - 1) // 2
        for j in range(i + 1, n):
            val = condensed[row_start + (j - i - 1)]
            W[i, j] = val
            W[j, i] = val
    return W
