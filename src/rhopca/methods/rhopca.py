import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.misc import standardize_array, generalized_eigen
from ..utils.covariance import compute_covariance
from ..utils.plotting import check_discrete, resolve_palette


class rhoPCA:
    """
    Contrastive PCA comparing a target group against a background group.

    Accepts a single AnnData object and splits it into target and background
    using a column in ``.obs``.

    Parameters
    ----------
    adata : AnnData
        AnnData object with normalized counts.
    contrast_column : str
        Column in ``adata.obs`` containing group labels.
    target : str
        Label for the target group.
    background : str
        Label for the background group.
    scale_variance : bool, default False
        Scale expression to unit variance in addition to zero-centering.
    n_components : int, optional
        Number of components to retain.  Defaults to all.
    """

    def __init__(
        self,
        adata,
        contrast_column,
        target,
        background,
        scale_variance=False,
        n_components=None,
    ):
        contrast_values = adata.obs[contrast_column].values

        for field in [target, background]:
            if field not in contrast_values:
                raise ValueError(
                    f"'{field}' is not in contrast column '{contrast_column}'."
                )

        self.adata = adata
        self.contrast_column = contrast_column
        self.target = target
        self.background = background
        self.target_label = target
        self.background_label = background
        self.scale_variance = scale_variance

        self.filt_target = contrast_values == target
        self.filt_background = contrast_values == background

        self.n_components = n_components if n_components is not None else adata.shape[1]

        # Store raw expression; scaling (if requested) happens in fit()
        self._target_X = adata[self.filt_target].X
        self._background_X = adata[self.filt_background].X

    def _get_adata(self, which):
        """Return the filtered AnnData view for 'target' or 'background'."""
        if which == 'target':
            return self.adata[self.filt_target]
        return self.adata[self.filt_background]

    def _get_obs(self, which, column):
        """Return obs column values for 'target' or 'background'."""
        return self._get_adata(which).obs[column].values

    @property
    def _var_index(self):
        return self.adata.var.index

    def _log(self, step, message):
        if self.verbose:
            print("-" * 40)
            print(f"\033[1m{step}\033[0m: {message}")

    def fit(self, method='schur', mu=None, bias=False, verbose=False):
        """
        Compute generalized eigendecomposition of target vs background covariance.

        Parameters
        ----------
        method : {'schur', 'tikhonov'}, default ``'schur'``
            Fallback solver when background covariance is singular.
        mu : float, optional
            Ridge parameter for Tikhonov regularization.
        bias : bool, default False
            Normalization for covariance estimation.  ``False`` divides by
            n-1; ``True`` divides by n.  Passed to :func:`compute_covariance`.
        """
        self.method = method

        X_t = standardize_array(self._target_X) if self.scale_variance else self._target_X
        X_b = standardize_array(self._background_X) if self.scale_variance else self._background_X

        if verbose: print("Computing target covariance...")
        Sigma_t = compute_covariance(X_t, bias=bias)
        if verbose: print("Computing background covariance...")
        Sigma_b = compute_covariance(X_b, bias=bias)
        if verbose: print("Covariance computation complete.")

        if verbose: print("Computing generalized eigenvectors and eigenvalues...")
        self.eigvals, self.eigvecs = generalized_eigen(
            Sigma_t, Sigma_b, method=method, mu=mu, n_components=self.n_components
        )
        if verbose: print("Eigendecomposition complete.")

        mu_t = np.asarray(X_t.mean(axis=0), dtype=np.float64).ravel()
        mu_b = np.asarray(X_b.mean(axis=0), dtype=np.float64).ravel()
        self.target_proj = np.asarray(X_t @ self.eigvecs) - mu_t @ self.eigvecs
        self.background_proj = np.asarray(X_b @ self.eigvecs) - mu_b @ self.eigvecs
        self.loadings = self.eigvecs * np.sqrt(self.eigvals)

    def get_top_genes(self, component=1, n_genes=5):
        """
        Return the top ``n_genes`` loading most strongly on a generalized eigenvector.

        Parameters
        ----------
        component : int
            Generalized eigenvector index (1-based).
        n_genes : int
            Number of top genes to return.
        """
        if component > self.loadings.shape[1]:
            raise ValueError(
                f"GE index out of bounds, max is {self.loadings.shape[1]}."
            )
        gene_scores = np.abs(self.loadings[:, component - 1])
        idx = np.argsort(gene_scores)[-n_genes:][::-1]
        return list(self._var_index[idx])

    def get_rhos(self, group_by=None):
        """
        Compute target/background variance ratios (rho) across all generalized
        eigenvectors, optionally grouped by a categorical variable in ``adata.obs``.

        Parameters
        ----------
        group_by : str, optional
            Column in ``adata.obs`` for per-group rho values.
        """
        n_components = self.loadings.shape[1]

        groups = (
            self.adata.obs[group_by].unique()
            if group_by is not None
            else np.array(['All'])
        )

        target_obs = self._get_adata('target').obs
        background_obs = self._get_adata('background').obs

        if group_by is not None:
            target_masks = {g: target_obs[group_by].values == g for g in groups}
            background_masks = {g: background_obs[group_by].values == g for g in groups}
        else:
            target_masks = {'All': np.ones(target_obs.shape[0], dtype=bool)}
            background_masks = {'All': np.ones(background_obs.shape[0], dtype=bool)}

        rhos = np.empty((len(groups), n_components))
        for i, group in enumerate(groups):
            t_var = np.var(self.target_proj[target_masks[group], :], axis=0, ddof=1)
            b_var = np.var(self.background_proj[background_masks[group], :], axis=0, ddof=1)
            rhos[i, :] = np.divide(
                t_var, b_var,
                out=np.full_like(t_var, np.nan),
                where=b_var > 0,
            )

        return pd.DataFrame(
            data=rhos,
            index=groups,
            columns=[f"GE {i + 1}" for i in range(n_components)],
        )

    # plotting ──────────────────────────────────────────────────────────────

    def plot(self, plot_type='scatter', components=(1, 2), *,
             color_by=None, palette=None):
        """
        Plot interface.

        Parameters
        ----------
        plot_type : {'scatter', 'hist'}, default ``'scatter'``
            Type of plot to produce.
            ``'scatter'`` — scatter of two generalized eigenvectors with target
                            and background overlaid.
            ``'hist'``    — KDE-smoothed histogram of one or more generalized
                            eigenvectors (one subplot per component).
        components : tuple of int, default ``(1, 2)``
            Generalized eigenvector indices to display (1-based).
            ``'scatter'``: first two entries used as x and y axes.
            ``'hist'``:    each entry produces one subplot.
        color_by : str, optional
            Column in ``.obs`` to color points / split histograms by
            (discrete columns only).
        palette : str or list, optional
            Colors to use for plotting.  When ``color_by`` is ``None``,
            the first two colors of the palette are used for target and
            background; defaults to steelblue and tomato.  When ``color_by``
            is set, one color per category is drawn from the palette; defaults
            to ``'tab10'``.  If a list is supplied that is too short, colors
            are padded from ``Set2``.
        """
        if isinstance(components, int):
            components = (components,)

        if plot_type == 'scatter':
            if len(components) < 2:
                raise ValueError("plot_type='scatter' requires at least 2 components.")
            self._plot_scatter(components[:2], color_by=color_by, palette=palette)
        elif plot_type == 'hist':
            self._plot_hist(components, color_by=color_by, palette=palette)
        else:
            raise ValueError(
                f"Unknown plot_type '{plot_type}'. "
                "Valid options for rhoPCA: 'scatter', 'hist'."
            )

    def _plot_scatter(self, components, *, color_by=None, palette=None):
        """
        Scatter of two generalized eigenvectors in side-by-side target and
        background plots.
        """
        comp_x, comp_y = components[0], components[1]
        n_avail = self.loadings.shape[1]
        if comp_x > n_avail or comp_y > n_avail:
            raise ValueError(f"GE index out of bounds (max={n_avail}).")

        x_col = f"GE {comp_x}"
        y_col = f"GE {comp_y}"

        target_df = pd.DataFrame({
            x_col: self.target_proj[:, comp_x - 1],
            y_col: self.target_proj[:, comp_y - 1],
        })
        background_df = pd.DataFrame({
            x_col: self.background_proj[:, comp_x - 1],
            y_col: self.background_proj[:, comp_y - 1],
        })

        fig, (ax_target, ax_background) = plt.subplots(1, 2, figsize=(12, 5))

        if color_by is not None:
            t_vals = self._get_obs('target', color_by)
            b_vals = self._get_obs('background', color_by)
            check_discrete(t_vals, color_by)
            target_df[color_by] = t_vals
            background_df[color_by] = b_vals
            categories = pd.Categorical(np.concatenate([t_vals, b_vals])).categories
            cat_colors = resolve_palette(palette or 'tab10', len(categories))
            scatter_palette = dict(zip(categories, cat_colors))

            sns.scatterplot(
                data=target_df, x=x_col, y=y_col,
                hue=color_by, palette=scatter_palette,
                ax=ax_target, s=40, alpha=0.8,
            )
            sns.scatterplot(
                data=background_df, x=x_col, y=y_col,
                hue=color_by, palette=scatter_palette,
                ax=ax_background, s=30, alpha=0.6,
            )
        else:
            pair = resolve_palette(palette, 2) if palette is not None else ['steelblue', 'tomato']
            sns.scatterplot(
                data=target_df, x=x_col, y=y_col,
                color=pair[0], ax=ax_target, s=40, alpha=0.8,
            )
            sns.scatterplot(
                data=background_df, x=x_col, y=y_col,
                color=pair[1], ax=ax_background, s=30, alpha=0.6,
            )

        all_x = np.concatenate([target_df[x_col].values, background_df[x_col].values])
        all_y = np.concatenate([target_df[y_col].values, background_df[y_col].values])
        x_buf = 0.05 * (all_x.max() - all_x.min())
        y_buf = 0.05 * (all_y.max() - all_y.min())
        x_lim = (all_x.min() - x_buf, all_x.max() + x_buf)
        y_lim = (all_y.min() - y_buf, all_y.max() + y_buf)

        ax_target.set_title(str(self.target_label), fontsize=14)
        ax_background.set_title(str(self.background_label), fontsize=14)

        for ax in (ax_target, ax_background):
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid(linestyle='--', color='lightgray', alpha=0.7)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title=color_by, bbox_to_anchor=(1.02, 0.5),
                      loc='center left')

        plt.tight_layout()
        plt.show()

    def _plot_hist(self, components, color_by=None, palette=None):
        """
        Histogram (with KDE) of target and background projections.

        When ``color_by`` is ``None``, one subplot per component is produced.
        Target and background use the first two colors of *palette* (defaults
        to steelblue and tomato).

        When ``color_by`` is set, the grid is ``n_components × n_categories``:
        each subplot shows target and background filtered to one category value.
        Target is colored by *palette* (defaults to ``'tab10'``); background
        is light gray (discrete columns only).

        Parameters
        ----------
        components : tuple of int
            Generalized eigenvector indices to plot (1-based).
        color_by : str, optional
            Column in ``.obs`` for per-category subplots.
        palette : str or list, optional
            Colors to use.  See :meth:`plot` for details.
        """
        n_avail = self.loadings.shape[1]
        for comp in components:
            if comp > n_avail:
                raise ValueError(f"GE index out of bounds, max is {n_avail}.")

        n_comps = len(components)

        if color_by is not None:
            cts = list(self._get_adata('target').obs[color_by].unique())
            check_discrete(self._get_obs('target', color_by), color_by)
            cat_colors = resolve_palette(palette or 'tab10', len(cts))
            n_cols = len(cts)

            fig, axes = plt.subplots(
                n_comps, n_cols,
                figsize=(5 * n_cols, 4 * n_comps),
                squeeze=False,
            )

            for row, comp in enumerate(components):
                for col, (ct, color) in enumerate(zip(cts, cat_colors)):
                    ax = axes[row, col]
                    filt_t = self._get_obs('target', color_by) == ct
                    filt_b = self._get_obs('background', color_by) == ct

                    t = self.target_proj[filt_t, comp - 1]
                    b = self.background_proj[filt_b, comp - 1]
                    rho = np.var(t) / np.var(b) if np.var(b) > 0 else np.nan

                    sns.histplot(b, kde=True, color='lightgray', stat='density',
                                 ax=ax, alpha=0.6, label=str(self.background_label))
                    sns.histplot(t, kde=True, color=color, stat='density',
                                 ax=ax, alpha=0.6, label=str(self.target_label))

                    ax.set_title(
                        f"GE {comp} — {ct}\n" + r"$\rho$ = " + f"{rho:.3f}"
                    )
                    ax.grid(alpha=0.2)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_ylabel('Density')
                    ax.legend()

        else:
            pair = resolve_palette(palette, 2) if palette is not None else ['steelblue', 'tomato']

            fig, axes = plt.subplots(n_comps, 1, figsize=(5, 4 * n_comps), squeeze=False)

            for row, comp in enumerate(components):
                ax = axes[row, 0]
                t = self.target_proj[:, comp - 1]
                b = self.background_proj[:, comp - 1]
                rho = np.var(t) / np.var(b) if np.var(b) > 0 else np.nan

                sns.histplot(t, kde=True, color=pair[0], stat='density',
                             ax=ax, alpha=0.6, label=str(self.target_label))
                sns.histplot(b, kde=True, color=pair[1], stat='density',
                             ax=ax, alpha=0.6, label=str(self.background_label))

                ax.set_title(f"GE {comp} distribution\n" + r"$\rho$ = " + f"{rho:.3f}")
                ax.grid(alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylabel('Density')
                ax.legend()

        plt.tight_layout()
        plt.show()
