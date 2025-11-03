import numpy as np
import pandas as pd 

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
        return list(genes[idx])


    def plot_scatter(self, x_GE=1, y_GE=2, color_by=None, palette="tab10"):
        """
        Scatter plot of two generalized eigenvectors for target and background
        using Seaborn for automatic coloring and legends.
    
        Parameters
        ----------
        x_GE, y_GE : int
            Indices of generalized eigenvectors to plot (1-based).
        color_by : str
            Column in adata.obs to color points by. If None, just use Target/Background.
        palette : str or list
            Seaborn palette name or list of colors.
        """
    
        # --- Bounds checking ---
        if x_GE > self.loadings.shape[1] or y_GE > self.loadings.shape[1]:
            raise ValueError(f"Selected GE exceeds calculated number of eigenvectors "
                             f"(max = {self.loadings.shape[1]}).")
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
    
        # --- Prepare DataFrames for Seaborn ---
        target_df = pd.DataFrame({
            f"GE{x_GE}": self.target_proj[:, x_GE-1],
            f"GE{y_GE}": self.target_proj[:, y_GE-1],
            "Group": "Target"
        })
    
        background_df = pd.DataFrame({
            f"GE{x_GE}": self.background_proj[:, x_GE-1],
            f"GE{y_GE}": self.background_proj[:, y_GE-1],
            "Group": "Background"
        })
    
        # If coloring by a column
        if color_by is not None:
            target_df[color_by] = self.adata[self.filt_target].obs[color_by].values
            background_df[color_by] = self.adata[self.filt_background].obs[color_by].values
    
        # Combine target and background
        plot_df = pd.concat([target_df, background_df], ignore_index=True)
    
        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
        # Target
        if color_by is not None:
            sns.scatterplot(
                data=plot_df[plot_df["Group"]=="Target"],
                x=f"GE{x_GE}", y=f"GE{y_GE}",
                hue=color_by,
                palette=palette,
                ax=axes[0],
                s=40,
                alpha=0.7
            )
        else:
            sns.scatterplot(
                data=plot_df[plot_df["Group"]=="Target"],
                x=f"GE{x_GE}", y=f"GE{y_GE}",
                hue="Group",
                palette={"Target": "red"},
                ax=axes[0],
                s=40,
                alpha=0.7,
                legend=False
            )
    
        axes[0].set_title("Target: " + str(self.target), fontsize=14)
        axes[0].grid(linestyle='--', color='lightgray', alpha=0.7)
    
        # Background
        if color_by is not None:
            sns.scatterplot(
                data=plot_df[plot_df["Group"]=="Background"],
                x=f"GE{x_GE}", y=f"GE{y_GE}",
                hue=color_by,
                palette=palette,
                ax=axes[1],
                s=40,
                alpha=0.7,
                legend=False  # Avoid duplicate legends
            )
        else:
            sns.scatterplot(
                data=plot_df[plot_df["Group"]=="Background"],
                x=f"GE{x_GE}", y=f"GE{y_GE}",
                hue="Group",
                palette={"Background": "blue"},
                ax=axes[1],
                s=40,
                alpha=0.7,
                legend=False
            )
    
        axes[1].set_title("Background: " + str(self.background), fontsize=14)
        axes[1].grid(linestyle='--', color='lightgray', alpha=0.7)
    
        # --- Legend ---
        if color_by is not None:
            handles, labels = axes[1].get_legend_handles_labels()
            fig.legend(handles, labels, title=color_by,
                       bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.0)
    
        plt.tight_layout()
        plt.show()



    def plot_hist(self, GE=1, color_by=None, colors=None, color=None):
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
            cts = [self.target]
            colors = [color or colors[0]]

        fig, ax = plt.subplots(len(cts), 1, figsize=(5, 4 * len(cts)), sharex=False)

        if len(cts) == 1:
            ax = [ax]

        for i, ct in enumerate(cts):
            if color_by is not None:
                filt_target = (self.adata[self.filt_target].obs[color_by].values == ct)
                filt_background = (self.adata[self.filt_background].obs[color_by].values == ct)
                title = f' - {ct}\n'
            else:
                filt_target = slice(None)
                filt_background = slice(None)
                title = f'\n'

            t = self.target_proj[filt_target, GE - 1]
            b = self.background_proj[filt_background, GE - 1]
            rho = np.var(t)/np.var(b)
            
            sns.histplot(t, kde=True,
                         color=colors[0], stat="density", ax=ax[i], alpha=0.6,label=self.target)
            sns.histplot(b, kde=True,
                         color="gray", stat="density", ax=ax[i], alpha=0.6,label=self.background)

            ax[i].grid(alpha=0.2)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_ylabel('Density')
            ax[i].legend()
            ax[i].set_title(f"GE {GE} distribution " + title + r" $\rho$ = " + f" {rho:.3f}")

        plt.tight_layout()
        plt.show()

    def get_rhos(self, group_by=None):
        """
        Compute target/background variance ratios (rho) across GEs,
        optionally grouped by a categorical variable in adata.obs.
        """
    
        n_GEs = self.loadings.shape[1]
        if group_by is not None:
            groups = self.adata.obs[group_by].unique()
        else:
            groups = np.array(['All'])
        n_groups = len(groups)
    

        target_obs = self.adata[self.filt_target].obs
        background_obs = self.adata[self.filt_background].obs
    
        if group_by is not None:
            target_masks = {g: (target_obs[group_by].values == g) for g in groups}
            background_masks = {g: (background_obs[group_by].values == g) for g in groups}
        else:
            target_masks = {'All': np.ones(target_obs.shape[0], dtype=bool)}
            background_masks = {'All': np.ones(background_obs.shape[0], dtype=bool)}
    
        rhos = np.empty((n_groups, n_GEs))
        for i, group in enumerate(groups):
            t_mask = target_masks[group]
            b_mask = background_masks[group]
    
            t_proj = self.target_proj[t_mask, :]
            b_proj = self.background_proj[b_mask, :]
    
            t_var = np.var(t_proj, axis=0, ddof=1)
            b_var = np.var(b_proj, axis=0, ddof=1)
    
            rho = np.divide(t_var, b_var, out=np.full_like(t_var, np.nan), where=b_var > 0)
            rhos[i, :] = rho
    
        df_ = pd.DataFrame(
            data=rhos,
            index=groups,
            columns=[f"GE {i+1}" for i in range(n_GEs)]
        )
    
        return df_
    

                
