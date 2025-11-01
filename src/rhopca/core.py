import numpy as np
from scipy.linalg import eigh, eig
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


def standardize_array(array):
    """
    Standardizes an array to zero mean and unit variance along columns.
    NaNs are replaced with 0.
    """
    standardized_array = (array - np.mean(array, axis=0)) / np.std(array, axis=0)
    return np.nan_to_num(standardized_array)


class rhoPCA:
    """
    rhoPCA contrastive dimensionality reduction comparing target vs background groups.

    Parameters
    ----------
    adata : AnnData
        AnnData object with normalized counts and .obs with contrast_column.
    contrast_column : str
        Column in adata.obs containing target and background labels.
    target : str
        Label for target group.
    background : str
        Label for background group.
    n_GEs : int, optional
        Number of generalized eigenvectors to keep. Defaults to all.
    """

    def __init__(self, adata, contrast_column, target, background, n_GEs=None):
        contrast_values = adata.obs[contrast_column].values

        for field in [target, background]:
            if field not in contrast_values:
                raise ValueError(f"'{field}' is not in contrast column '{contrast_column}'.")

        self.adata = adata
        self.contrast_column = contrast_column
        self.target = target
        self.background = background

        self.filt_target = contrast_values == target
        self.filt_background = contrast_values == background

        self.n_GEs = n_GEs if n_GEs is not None else adata.shape[1]

    def fit(self):
        """Compute generalized eigen decomposition of target vs background covariance."""

        # Extract counts
        target_counts = self.adata[self.filt_target].X.toarray()
        background_counts = self.adata[self.filt_background].X.toarray()

        # Standardize
        target_std = standardize_array(target_counts)
        background_std = standardize_array(background_counts)

        # Covariance matrices
        Sigma_t = np.cov(target_std, rowvar=False)
        Sigma_b = np.cov(background_std, rowvar=False)

        # Generalized eigen decomposition
        try:
            eigvals_rq, eigvecs_rq = eigh(Sigma_t, Sigma_b)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Covariance matrix of background is not positive definite: "
                "falling back to eig, keeping real positive eigenvalues."
            )
            eigvals_rq, eigvecs_rq = eig(Sigma_t, Sigma_b)

        # Keep only finite, positive eigenvalues
        filt = (eigvals_rq > 0) & (np.isfinite(eigvals_rq))
        eigvals_rq = eigvals_rq[filt]
        eigvecs_rq = eigvecs_rq[:, filt]

        # Sort in descending order
        idx = np.argsort(eigvals_rq)[::-1]
        eigvals_rq = eigvals_rq[idx]
        eigvecs_rq = eigvecs_rq[:, idx]

        # Keep only top n_GEs
        self.eigvals = eigvals_rq[:self.n_GEs]
        self.eigvecs = eigvecs_rq[:, :self.n_GEs]

        # Project data
        self.target_proj = target_std @ self.eigvecs
        self.background_proj = background_std @ self.eigvecs

        # Loadings
        self.loadings = self.eigvecs * np.sqrt(self.eigvals)

    def get_top_genes(self, GE=1, n_genes=5):
        """
        Return the top n_genes for a generalized eigenvector.

        Parameters
        ----------
        GE : int
            Generalized eigenvector index (1-based).
        n_genes : int
            Number of top genes to return.
        """
        if GE > self.loadings.shape[1]:
            raise ValueError(f"Selected GE out of bounds, max is {self.loadings.shape[1]}")

        genes = self.adata.var.index
        gene_scores = np.abs(self.loadings[:, GE - 1])
        idx = np.argsort(gene_scores)[-n_genes:][::-1]  # descending order
        return genes[idx]

    def plot_scatter(self, x_GE=1, y_GE=2, color_by=None, colors=None):
        """
        Scatter plot of two generalized eigenvectors for target and background.

        Parameters
        ----------
        x_GE, y_GE : int
            Indices of generalized eigenvectors to plot (1-based).
        color_by : str
            Column in adata.obs to color points by.
        colors : list or array
            Colors to use if color_by is None.
        """
        if x_GE > self.loadings.shape[1] or y_GE > self.loadings.shape[1]:
            raise ValueError("Selected GE exceeds calculated number of eigenvectors.")

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Color assignment
        if color_by is not None:
            target_colors = self.adata[self.filt_target].obs[color_by].values
            background_colors = self.adata[self.filt_background].obs[color_by].values
        elif colors is not None:
            target_colors = colors[0]
            background_colors = colors[1]
        else:
            target_colors = 'red'
            background_colors = 'blue'

        ax[0].scatter(self.target_proj[:, x_GE - 1], self.target_proj[:, y_GE - 1], c=target_colors)
        ax[0].set_xlabel(f'GE {x_GE}', fontsize=12)
        ax[0].set_ylabel(f'GE {y_GE}', fontsize=12)
        ax[0].grid(linestyle='--', color='gray')

        ax[1].scatter(self.background_proj[:, x_GE - 1], self.background_proj[:, y_GE - 1], c=background_colors)
        ax[1].set_xlabel(f'GE {x_GE}', fontsize=12)
        ax[1].set_ylabel(f'GE {y_GE}', fontsize=12)
        ax[1].grid(linestyle='--', color='gray')

        plt.tight_layout()
        plt.show()

    def plot_hist(self, GE=1, color_by=None, colors=None):
        """
        Plot histograms (with KDE) of target and background projections for a GE.

        Parameters
        ----------
        GE : int
            Generalized eigenvector index (1-based).
        color_by : str
            Column in adata.obs for coloring.
        colors : list
            List of colors to use for each unique value in color_by.
        """
        if GE > self.loadings.shape[1]:
            raise ValueError(f"Selected GE out of bounds, max is {self.loadings.shape[1]}")

        # Default colors
        if colors is None:
            colors = sns.color_palette("Set2", n_colors=2)

        # Get unique categories
        if color_by is not None:
            cts = self.adata.obs[color_by].unique()
        else:
            cts = [self.target, self.background]

        fig, ax = plt.subplots(len(cts), 1, figsize=(5, 4 * len(cts)), sharex=False)

        if len(cts) == 1:
            ax = [ax]

        for i, ct in enumerate(cts):
            if color_by is not None:
                filt_target = (self.adata[self.filt_target].obs[color_by].values == ct)
                filt_background = (self.adata[self.filt_background].obs[color_by].values == ct)
            else:
                filt_target = slice(None)
                filt_background = slice(None)

            sns.histplot(self.target_proj[filt_target, GE - 1], kde=True,
                         color=colors[0], stat="density", ax=ax[i], alpha=0.6)
            sns.histplot(self.background_proj[filt_background, GE - 1], kde=True,
                         color=colors[1], stat="density", ax=ax[i], alpha=0.6)

            ax[i].grid(alpha=0.2)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_ylabel('Density')
            ax[i].set_title(f"GE {GE} distribution — {ct}")

        plt.tight_layout()
        plt.show()
