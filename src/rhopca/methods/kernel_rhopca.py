import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt

from .rhopca import rhoPCA
from ..utils.misc import standardize_array, generalized_eigen
from ..utils.covariance import (
    DEFAULT_N_JOBS, _n_threads, compute_covariance, weighted_covariance,
)
from ..utils.plotting import resolve_continuous_palette


class kernelRhoPCA(rhoPCA):
    """
    Kernel-weighted contrastive PCA (k-rhoPCA).

    Extends :class:`rhoPCA` to support a spatial (or other pairwise) kernel
    applied to the target covariance, and optionally to the background.

    Parameters
    ----------
    adata_target : AnnData
        Target group expression data.
    adata_background : AnnData or ``'Identity'``
        Background group expression data.  Must share the same gene set as
        ``adata_target``.  Pass ``'Identity'`` to use the identity matrix as the
        background covariance, which reduces the method to a standard
        eigendecomposition of the (kernel-weighted) target covariance.
    coordinates_key : str, optional
        Key in ``.obsm`` holding spatial (or other) coordinates used to compute
        pairwise distances for the kernel.  Prompted if not provided.
    kernel : str, callable, or None, optional
        Kernel to apply when computing the target covariance.
        Built-in options: ``'gaussian'``, ``'inverse_distance'``, ``'wendland'``.
        The kernel should be positive semi-definite so that the weighted
        covariance is PSD.
        A callable ``f(distances, **kwargs) -> weights`` is also accepted.
        Pass ``None`` to use a standard (unweighted) covariance.
    kernel_kwargs : dict, optional
        Kernel parameters:

        * ``bandwidth`` — ``'gaussian'`` width σ: how quickly weights decay with
          distance (never exactly zero).  Defaults to the median pairwise
          distance, which is on the scale of the whole dataset.
        * ``buffer`` — ``'inverse_distance'`` offset in ``1 / (d + buffer)``.
        * ``support`` — distance beyond which weights are exactly zero.
          Required for ``'wendland'``; optional for other kernels, which are
          then tapered by a Wendland kernel.  Giving a support makes the
          weight matrix sparse, which large datasets need; about
          10 × bandwidth keeps a Gaussian close to its untapered shape.

        With 100,000 or more cells and no ``support``, a Gaussian uses
        ``support = 10 × bandwidth`` automatically (with a warning); other
        kernels raise an error.
    background_kernel : bool, default False
        Apply ``kernel`` to the background covariance as well.
    scale_variance : bool, default False
        Scale expression to unit variance in addition to zero-centering.
    n_components : int, optional
        Number of generalized eigenvectors to retain.  Defaults to all valid components.
    """

    def __init__(
        self,
        adata_target,
        adata_background,
        *,
        coordinates_key=None,
        scale_variance=False,
        n_components=None,
        kernel=None,
        kernel_kwargs=None,
        background_kernel=False,
    ):
        self.identity_background = isinstance(adata_background, str) and adata_background == "Identity"

        # --- validate / align gene sets ---
        if not self.identity_background:
            if not adata_target.var_names.equals(adata_background.var_names):
                common = adata_target.var_names.intersection(adata_background.var_names)
                only_t = len(adata_target.var_names) - len(common)
                only_b = len(adata_background.var_names) - len(common)
                warnings.warn(
                    f"Gene sets do not match. Filtering to {len(common)} common genes "
                    f"({only_t} dropped from target, {only_b} dropped from background). "
                )
                adata_target = adata_target[:, common]
                adata_background = adata_background[:, common]

        self.adata_target = adata_target
        self.adata_background = adata_background
        self.target_label = 'Target'
        self.background_label = 'Background'
        self.scale_variance = scale_variance
        self.background_kernel = background_kernel
        self.n_components = n_components if n_components is not None else adata_target.shape[1]
        self.kernel_kwargs = kernel_kwargs or {}

        # --- prompt for coordinates_key ---
        coordinates_key = coordinates_key if coordinates_key else 'spatial'
        self.coordinates_key = coordinates_key

        # --- prompt for kernel ---
        if kernel is None:
            warnings.warn(
                "kernel=None: no kernel will be applied to the target covariance. "
                "Pass kernel='gaussian' or another kernel to enable kernel weighting."
            )
        self.kernel = kernel

        # Store raw expression; scaling (if requested) happens in fit()
        self._target_X = adata_target.X
        self._background_X = None if self.identity_background else adata_background.X

        # --- store spatial coordinates (None if key is absent) ---
        self._target_coords = (
            np.asarray(adata_target.obsm[coordinates_key], dtype=np.float64)
            if coordinates_key in adata_target.obsm else None
        )
        if self.identity_background:
            self._background_coords = None
        else:
            self._background_coords = (
                np.asarray(adata_background.obsm[coordinates_key], dtype=np.float64)
                if coordinates_key in adata_background.obsm else None
            )


    @property
    def target_only(self):
        """``True`` when ``adata_background='Identity'`` (no background projection)."""
        return self.identity_background

    def _get_adata(self, which):
        """Return AnnData for 'target' or 'background'."""
        return self.adata_target if which == 'target' else self.adata_background

    def _get_obs(self, which, column):
        """Return obs column values for 'target' or 'background'."""
        return self._get_adata(which).obs[column].values

    @property
    def _var_index(self):
        return self.adata_target.var.index

    def _has_target_coordinates(self):
        """Return True when target AnnData has the coordinate key in .obsm."""
        return self._target_coords is not None


    def fit(self, *, method='schur', mu=None, bias=False, normalize_by_weights=False,
            verbose=False, n_jobs=DEFAULT_N_JOBS):
        """
        Compute the (kernel-weighted) generalized eigendecomposition.

        Parameters
        ----------
        method : {'schur', 'tikhonov'}, default ``'schur'``
            Fallback solver when the background covariance is singular.
            Ignored when ``adata_background='Identity'``.
        mu : float, optional
            Ridge for Tikhonov regularization.  Defaults to
            ``1e-6 · trace(Σ_b) / n_features``.
            Ignored when ``adata_background='Identity'``.
        bias : bool, default False
            Normalization for covariance estimation.  ``False`` divides by
            n-1; ``True`` divides by n.  Passed to :func:`compute_covariance`.
        normalize_by_weights : bool, default False
            For kernel-weighted covariances, divide by ``S0 = Σ_ij W_ij`` (as in
            Moran's I) instead of n-1 / n, controlling for the total weight in
            ``W``.  Unweighted covariances (e.g. the background when
            ``background_kernel=False``) still follow ``bias``.  Only rescales
            the eigenvalues, loadings, and rhos; eigenvectors are unchanged.
        n_jobs : int, default 16
            Number of threads for the neighbor search, kernel evaluation,
            covariance products, eigendecomposition, and projections (and
            later for :meth:`get_rhos`).  ``-1`` uses all cores.  Capped at
            the number of cores available.
        """
        self.method = method
        self.verbose = verbose
        self.bias = bias
        self.normalize_by_weights = normalize_by_weights
        self.n_jobs = n_jobs

        X_t = standardize_array(self._target_X) if self.scale_variance else self._target_X

        # Warn if kernel requested but coordinates are missing
        if self.kernel is not None and self._target_coords is None:
            warnings.warn(
                f"Kernel '{self.kernel}' requested but '{self.coordinates_key}' "
                "not found in target .obsm. Falling back to standard covariance."
            )
        if self.background_kernel and self._background_coords is None:
            warnings.warn(
                f"background_kernel=True but '{self.coordinates_key}' "
                "not found in background .obsm. Falling back to standard covariance."
            )

        target_kernel = self.kernel if self._target_coords is not None else None
        use_background_kernel = self.background_kernel and self._background_coords is not None

        kernel_str = (
            f"kernel='{self.kernel}'" + (f", {self.kernel_kwargs}" if self.kernel_kwargs else "")
            if target_kernel is not None else "no kernel"
        )
        self._log("Covariance", f"Computing target covariance ({kernel_str})...")
        Sigma_t, self.W_target_ = compute_covariance(
            X_t,
            kernel=target_kernel,
            bias=bias,
            normalize_by_weights=normalize_by_weights,
            return_weights=True,
            n_jobs=n_jobs,
            coordinates=self._target_coords if target_kernel is not None else None,
            kernel_kwargs=self.kernel_kwargs,
        )

        if self.identity_background:
            self._log("Covariance", "Background is Identity — skipping background covariance.")
            Sigma_b = None
            self.W_background_ = None
        else:
            X_b = standardize_array(self._background_X) if self.scale_variance else self._background_X
            background_kernel = self.kernel if use_background_kernel else None
            bg_kernel_str = kernel_str if background_kernel is not None else "no kernel"
            self._log("Covariance", f"Computing background covariance ({bg_kernel_str})...")
            Sigma_b, self.W_background_ = compute_covariance(
                X_b,
                kernel=background_kernel,
                bias=bias,
                normalize_by_weights=normalize_by_weights,
                return_weights=True,
                n_jobs=n_jobs,
                coordinates=self._background_coords if background_kernel is not None else None,
                kernel_kwargs=self.kernel_kwargs,
            )

        self._log("Covariance", "Complete.")

        with _n_threads(n_jobs):
            self._log("Eigendecomposition", "Computing eigenvectors and eigenvalues...")
            self.eigvals, self.eigvecs = generalized_eigen(
                Sigma_t, Sigma_b, method=method, mu=mu, n_components=self.n_components
            )
            self._log("Eigendecomposition", "Complete.")

            mu_t = np.asarray(X_t.mean(axis=0), dtype=np.float64).ravel()

            self._log("Projecting", "...")
            self.target_proj = np.asarray(X_t @ self.eigvecs) - mu_t @ self.eigvecs

            if self.identity_background:
                self.background_proj = None
            else:
                mu_b = np.asarray(X_b.mean(axis=0), dtype=np.float64).ravel()
                self.background_proj = np.asarray(X_b @ self.eigvecs) - mu_b @ self.eigvecs

        self._log("", "Finished computing projections.")
        self.loadings = self.eigvecs * np.sqrt(np.abs(self.eigvals))
        

    def get_rhos(self, *, group_by=None):
        """
        Compute target-to-background variance ratios (rho) across all components.

        Rho is the ratio that kernel rhoPCA maximizes: for component ``v``,
        ``rho = vᵀ Σ_t^W v / vᵀ Σ_b v``, using the same weight matrices and
        normalization (``bias``, ``normalize_by_weights``) as :meth:`fit`.
        For the full target this equals the eigenvalues.  Per-group target
        variances restrict ``W`` to the cells in that group.

        When ``adata_background='Identity'``, there is no background projection
        so rho is undefined.  The eigenvalues of the target covariance are
        returned instead (``group_by`` is ignored).

        Parameters
        ----------
        group_by : str, optional
            Column in target ``.obs`` for per-group rho values.  Each target
            group is compared against the full background.
        """
        n_components = self.loadings.shape[1]
        columns = [f"GE {j + 1}" for j in range(n_components)]

        if self.target_only:
            return pd.DataFrame(
                data=self.eigvals[:n_components].reshape(1, -1),
                index=['Eigenvalue'],
                columns=columns,
            )


        groups = (
            self.adata_target.obs[group_by].unique()
            if group_by is not None
            else np.array(['All'])
        )

        rhos = np.empty((len(groups), n_components))

        for i, group in enumerate(groups):
            t_mask = (
                self._get_obs('target', group_by) == group
                if group_by is not None
                else slice(None)
            )
            t_var = self._weighted_proj_var(self.target_proj, self.W_target_, t_mask)
            b_var = self._weighted_proj_var(self.background_proj, self.W_background_)
            rhos[i, :] = np.divide(
                t_var, b_var,
                out=np.full_like(t_var, np.nan),
                where=b_var > 0,
            )

        return pd.DataFrame(data=rhos, index=groups, columns=columns)

    def _weighted_proj_var(self, proj, W, mask=slice(None)):
        """Per-component (kernel-weighted) variance of projections, as in fit."""
        if W is not None and not isinstance(mask, slice):
            idx = np.flatnonzero(mask)
            W = W[idx][:, idx] if hasattr(W, 'tocsr') else W[np.ix_(idx, idx)]
        cov = weighted_covariance(
            proj[mask, :], W,
            bias=self.bias,
            normalize_by_weights=self.normalize_by_weights,
            n_jobs=self.n_jobs,
        )
        return np.diag(cov)

    # _plot_scatter and _plot_hist are inherited from rhoPCA.

    def plot(self, plot_type='scatter', components=(1, 2), *,
             color_by=None, palette=None):
        """
        Plot interface.  Extends :meth:`rhoPCA.plot` with
        ``plot_type='spatial'``.

        Parameters
        ----------
        plot_type : {'scatter', 'hist', 'spatial'}, default ``'scatter'``
            Type of plot to produce.
            ``'scatter'`` — Scatter plot of two generalized eigenvectors.
            ``'hist'``    — KDE-smoothed histogram of one or more generalized
                            eigenvectors (one subplot per component).
            ``'spatial'`` — one spatial map per component (requires coordinates).
                            Always uses a continuous colormap; if *palette* is a
                            list it is ignored and ``'viridis'`` is used instead.
        components : tuple of int, default ``(1, 2)``
            Generalized eigenvector indices to display (1-based).
        color_by : str, optional
            Column in ``.obs`` to color points / split histograms by
            (scatter and hist modes only; discrete columns only).
        palette : str or list
            Seaborn palette for scatter/hist or colormap name for spatial.
            Lists are accepted for scatter/hist and padded with ``Set2`` if
            too short.
        """
        if plot_type == 'spatial':
            if not self._has_target_coordinates():
                raise ValueError(
                    f"plot_type='spatial' requires '{self.coordinates_key}' "
                    "in target .obsm."
                )
            self._plot_spatial(components, palette=palette)
        else:
            super().plot(plot_type, components,
                         color_by=color_by, palette=palette)

    def _plot_spatial(self, components, palette='viridis'):
        """Spatial maps: x/y = coordinates, color = component projection value.

        Always uses a continuous colormap.  If *palette* is a list it is
        replaced with the default continuous palette (``'viridis'``).
        """
        cmap = resolve_continuous_palette(palette)
        t_coords = self._target_coords
        bg_has_coords = self._background_coords is not None

        n_cols = len(components)
        n_rows = 2 if bg_has_coords else 1
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
            squeeze=False,
        )

        for col, comp in enumerate(components):
            comp_idx = comp - 1

            sc = axes[0, col].scatter(
                t_coords[:, 0], t_coords[:, 1],
                c=self.target_proj[:, comp_idx],
                cmap=cmap, s=8, alpha=0.8, rasterized=True,
            )
            axes[0, col].set_title(
                f"{self.target_label} — GE {comp}", fontsize=12
            )
            axes[0, col].set_aspect('equal')
            axes[0, col].invert_yaxis()
            axes[0, col].axis('off')
            plt.colorbar(sc, ax=axes[0, col], shrink=0.7)

            if bg_has_coords:
                bg_coords = self._background_coords
                sc2 = axes[1, col].scatter(
                    bg_coords[:, 0], bg_coords[:, 1],
                    c=self.background_proj[:, comp_idx],
                    cmap=cmap, s=8, alpha=0.8, rasterized=True,
                )
                axes[1, col].set_title(
                    f"{self.background_label} — GE {comp}", fontsize=12
                )
                axes[1, col].set_aspect('equal')
                axes[1, col].invert_yaxis()
                axes[1, col].axis('off')
                plt.colorbar(sc2, ax=axes[1, col], shrink=0.7)

        plt.tight_layout()
        plt.show()
