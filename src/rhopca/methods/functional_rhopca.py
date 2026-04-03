from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import seaborn as sns
from anndata import AnnData

from .rhopca import rhoPCA
from ..utils.covariance import compute_covariance
from ..utils.misc import generalized_eigen, standardize_array
from ..utils.plotting import resolve_palette


def _require_skfda():
    try:
        import skfda
        from skfda.representation.basis import BSplineBasis, FourierBasis, MonomialBasis
        from skfda.representation.irregular import FDataIrregular
    except ImportError as exc:
        raise ImportError(
            "functionalRhoPCA requires the optional dependency 'scikit-fda'. "
            "Install it with `pip install scikit-fda`."
        ) from exc

    return skfda, FDataIrregular, BSplineBasis, FourierBasis, MonomialBasis


# +
class functionalRhoPCA(rhoPCA):
    """
    Functional rhoPCA on a single time-varying feature.

    The input may be either:

    * an ``AnnData`` object plus ``gene=<var_name>``, or
    * a ``pandas.DataFrame`` containing the observed values directly.

    Repeated rows for the same observation are interpreted as irregularly
    sampled time points for one underlying function.
    """

    def __init__(
        self,
        data,
        contrast_column,
        target,
        background,
        *,
        observation,
        time,
        gene=None,
        basis="bspline",
        num_basis_functions=5,
        scale_variance=False,
        n_components=None,
    ):
        if not isinstance(data, (AnnData, pd.DataFrame)):
            raise TypeError("`data` must be an AnnData object or a pandas DataFrame.")
        if isinstance(data, AnnData) and gene is None:
            raise ValueError("`gene` is required when `data` is an AnnData object.")
        if num_basis_functions < 1:
            raise ValueError("`num_basis_functions` must be at least 1.")

        self.adata = data if isinstance(data, AnnData) else None
        self.data = data
        self.gene = gene
        self.contrast_column = contrast_column
        self.target = target
        self.background = background
        self.target_label = target
        self.background_label = background
        self.observation = observation
        self.time = time
        self.basis_name = basis
        self.num_basis_functions = num_basis_functions
        self.scale_variance = scale_variance
        self.n_components = n_components if n_components is not None else num_basis_functions
        self.value_column = "__functional_value__"

        self.long_df = self._prepare_long_dataframe(data)
        self._validate_long_dataframe()

        self.time_points_ = np.sort(self.long_df[self.time].unique())
        self.obs_ = self._summarize_observations(self.long_df)

        contrast_values = self.obs_[self.contrast_column].values
        self.filt_target = contrast_values == target
        self.filt_background = contrast_values == background

        if self.filt_target.sum() < 2:
            raise ValueError("At least two target observations are required.")
        if self.filt_background.sum() < 2:
            raise ValueError("At least two background observations are required.")

        self._target_obs_df = self.obs_.loc[self.filt_target].reset_index(drop=True)
        self._background_obs_df = self.obs_.loc[self.filt_background].reset_index(drop=True)

    @property
    def target_only(self):
        return False

    @property
    def _var_index(self):
        return pd.Index([self.gene] if self.gene is not None else [self.value_column])

    def _get_adata(self, which):
        obs = self._target_obs_df if which == "target" else self._background_obs_df
        return SimpleNamespace(obs=obs)

    def _get_obs(self, which, column):
        return self._get_adata(which).obs[column].values

    def _prepare_long_dataframe(self, data):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if self.gene is not None:
                if self.gene not in df.columns:
                    raise ValueError(f"`gene='{self.gene}'` is not a column in the DataFrame.")
                value_column = self.gene
            else:
                reserved = {self.observation, self.time, self.contrast_column}
                candidates = [c for c in df.columns if c not in reserved]
                if len(candidates) != 1:
                    raise ValueError(
                        "For DataFrame input, provide `gene=<column_name>` or pass a "
                        "DataFrame with exactly one value column besides observation/time/contrast."
                    )
                value_column = candidates[0]
            df[self.value_column] = pd.to_numeric(df[value_column], errors="raise")
            return df

        if self.gene not in data.var.index:
            raise ValueError(f"`gene='{self.gene}'` is not present in `adata.var.index`.")

        gene_view = data[:, self.gene].X
        if scipy.sparse.issparse(gene_view):
            gene_values = gene_view.toarray().ravel()
        else:
            gene_values = np.asarray(gene_view).ravel()

        df = data.obs.copy()
        df[self.value_column] = gene_values
        return df

    def _validate_long_dataframe(self):
        required = [self.observation, self.time, self.contrast_column, self.value_column]
        missing = [col for col in required if col not in self.long_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}.")

        if self.long_df[required].isnull().any().any():
            raise ValueError(
                "Missing values are not allowed in observation, time, contrast, or value columns."
            )

        self.long_df[self.time] = pd.to_numeric(self.long_df[self.time], errors="raise")
        self.long_df[self.value_column] = pd.to_numeric(
            self.long_df[self.value_column], errors="raise"
        )
        self.long_df = self.long_df.sort_values([self.observation, self.time]).reset_index(drop=True)

        contrast_values = self.long_df[self.contrast_column].unique()
        for field in [self.target, self.background]:
            if field not in contrast_values:
                raise ValueError(
                    f"'{field}' is not in contrast column '{self.contrast_column}'."
                )

        label_counts = self.long_df.groupby(self.observation)[self.contrast_column].nunique()
        if (label_counts > 1).any():
            bad_obs = label_counts[label_counts > 1].index.tolist()[:5]
            raise ValueError(
                "Each observation must map to exactly one contrast label. "
                f"Found conflicting labels for observations such as {bad_obs}."
            )

    def _summarize_observations(self, df):
        return df.groupby(self.observation, sort=False).first().reset_index()

    def _build_basis(self, domain_range):
        _, _, BSplineBasis, FourierBasis, MonomialBasis = _require_skfda()
        basis_key = self.basis_name.lower().replace("-", "").replace("_", "")

        if basis_key in {"bspline", "spline"}:
            return BSplineBasis(domain_range=domain_range, n_basis=self.num_basis_functions)
        if basis_key in {"fourier"}:
            return FourierBasis(domain_range=domain_range, n_basis=self.num_basis_functions)
        if basis_key in {"monomial", "polynomial"}:
            return MonomialBasis(domain_range=domain_range, n_basis=self.num_basis_functions)

        raise ValueError("`basis` must be one of {'bspline', 'fourier', 'monomial'}.")

    def _build_irregular_fd(self, df, fd_cls):
        points = []
        values = []
        for _, group in df.groupby(self.observation, sort=False):
            t = group[self.time].to_numpy(dtype=np.float64)
            y = group[self.value_column].to_numpy(dtype=np.float64)
            points.append(t)
            values.append(y[:, np.newaxis])

        lengths = [len(t) for t in points]
        start_indices = np.cumsum([0] + lengths[:-1], dtype=int)

        return fd_cls(
            start_indices=start_indices,
            points=np.concatenate(points),
            values=np.vstack(values),
        )

    def _coefficients_from_fd(self, fd, basis):
        fd_basis = fd.to_basis(basis)
        coefficients = np.asarray(fd_basis.coefficients, dtype=np.float64)
        if coefficients.ndim == 3 and coefficients.shape[-1] == 1:
            coefficients = coefficients[..., 0]
        return coefficients

    def get_top_genes(self, component=1, n_genes=5):
        raise NotImplementedError(
            "functionalRhoPCA models one functional signal at a time; `get_top_genes` is unavailable."
        )

    def plot(
        self,
        plot_type="scatter",
        components=(1, 2),
        *,
        palette=None,
        time_values=None,
        n_time_points=200,
    ):
        """
        Plot interface for functional rhoPCA.

        Parameters
        ----------
        plot_type : {'scatter', 'hist', 'eigenfunction'}, default ``'scatter'``
            ``'scatter'`` plots two eigenfunction scores against each other.
            ``'hist'`` delegates to the base rhoPCA histogram interface.
            ``'eigenfunction'`` plots one fitted eigenfunction in time.
        components : int or tuple of int, default ``(1, 2)``
            Component index or indices (1-based).
        palette : str or list, optional
            Color palette. For score scatter, the first two colors are used
            for target and background. For eigenfunction plots, the first
            color is used for the curve.
        time_values : array-like, optional
            Time grid used for evaluating the eigenfunction. If omitted,
            a dense inclusive grid on ``[min(time), max(time)]`` is used.
        n_time_points : int, default 200
            Number of time points for the default grid.
        """
        if plot_type == "eigenfunction":
            component = components[0] if isinstance(components, tuple) else components
            self._plot_eigenfunction(
                component=component,
                time_values=time_values,
                palette=palette,
                n_time_points=n_time_points,
            )
            return

        super().plot(plot_type=plot_type, components=components, palette=palette)

    def _plot_scatter(self, components, *, color_by=None, palette=None):
        """
        Scatter plot of two eigenfunction scores with target/background colors.

        ``color_by`` is intentionally unsupported for functional rhoPCA.
        """
        if color_by is not None:
            raise ValueError("`color_by` is not supported for functionalRhoPCA scatter plots.")

        comp_x, comp_y = components[0], components[1]
        n_avail = self.target_proj.shape[1]
        if comp_x > n_avail or comp_y > n_avail:
            raise ValueError(f"GE index out of bounds (max={n_avail}).")

        x_col = f"GE {comp_x}"
        y_col = f"GE {comp_y}"
        target_df = pd.DataFrame(
            {
                x_col: self.target_proj[:, comp_x - 1],
                y_col: self.target_proj[:, comp_y - 1],
            }
        )
        background_df = pd.DataFrame(
            {
                x_col: self.background_proj[:, comp_x - 1],
                y_col: self.background_proj[:, comp_y - 1],
            }
        )

        pair = resolve_palette(palette, 2) if palette is not None else ["steelblue", "tomato"]

        fig, (ax_target, ax_background) = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(
            data=target_df,
            x=x_col,
            y=y_col,
            color=pair[0],
            ax=ax_target,
            s=40,
            alpha=0.8,
        )
        sns.scatterplot(
            data=background_df,
            x=x_col,
            y=y_col,
            color=pair[1],
            ax=ax_background,
            s=30,
            alpha=0.6,
        )

        all_x = np.concatenate([target_df[x_col].values, background_df[x_col].values])
        all_y = np.concatenate([target_df[y_col].values, background_df[y_col].values])
        x_buf = 0.05 * (all_x.max() - all_x.min()) if all_x.max() > all_x.min() else 1.0
        y_buf = 0.05 * (all_y.max() - all_y.min()) if all_y.max() > all_y.min() else 1.0
        x_lim = (all_x.min() - x_buf, all_x.max() + x_buf)
        y_lim = (all_y.min() - y_buf, all_y.max() + y_buf)

        for ax, title in zip(
            [ax_target, ax_background],
            [str(self.target_label), str(self.background_label)],
        ):
            ax.set_title(title, fontsize=14)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.grid(linestyle="--", color="lightgray", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def _plot_eigenfunction(self, component=1, *, time_values=None, palette=None, n_time_points=200):
        """
        Plot one fitted eigenfunction on a supplied or default time grid.
        """
        if component < 1 or component > len(self.eigenfunction_dict):
            raise ValueError(
                f"GE index out of bounds (max={len(self.eigenfunction_dict)})."
            )

        if time_values is None:
            time_values = np.linspace(
                self.time_points_.min(),
                self.time_points_.max(),
                n_time_points,
                endpoint=True,
                dtype=np.float64,
            )
        else:
            time_values = np.asarray(time_values, dtype=np.float64)

        if time_values.ndim != 1 or time_values.size == 0:
            raise ValueError("`time_values` must be a non-empty one-dimensional array.")

        curve_fd = self.eigenfunction_dict[component]
        curve = np.asarray(curve_fd(time_values), dtype=np.float64)
        if curve.ndim == 3 and curve.shape[-1] == 1:
            curve = curve[0, :, 0]
        elif curve.ndim == 2 and curve.shape[0] == 1:
            curve = curve[0]
        else:
            curve = np.ravel(curve)

        color = resolve_palette(palette, 1)[0] if palette is not None else "steelblue"

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(time_values, curve, color=color, linewidth=2)
        ax.axhline(0, color="lightgray", linewidth=1, linestyle="--")
        ax.set_title(f"GE {component} eigenfunction", fontsize=14)
        ax.set_xlabel(self.time)
        ax.set_ylabel("Eigenfunction value")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    def fit(self, method="schur", mu=None, bias=False, verbose=False, eps=1e-8):
        self.method = method
        self.verbose = verbose
        self.bias = bias

        skfda, FDataIrregular, _, _, _ = _require_skfda()

        domain_range = ((self.time_points_.min(), self.time_points_.max()),)
        self.basis = self._build_basis(domain_range)

        target_df = self.long_df.loc[self.long_df[self.contrast_column] == self.target].copy()
        background_df = self.long_df.loc[
            self.long_df[self.contrast_column] == self.background
        ].copy()

        self.fd_irregular = self._build_irregular_fd(self.long_df, FDataIrregular)
        self.fd_target_irregular = self._build_irregular_fd(target_df, FDataIrregular)
        self.fd_background_irregular = self._build_irregular_fd(background_df, FDataIrregular)

        A_t = self._coefficients_from_fd(self.fd_target_irregular, self.basis)
        A_b = self._coefficients_from_fd(self.fd_background_irregular, self.basis)


        if self.scale_variance:
            A_t = np.asarray(standardize_array(A_t), dtype=np.float64)
            A_b = np.asarray(standardize_array(A_b), dtype=np.float64)

#         self._target_X = A_t
#         self._background_X = A_b
        self.target_coefficients = A_t
        self.background_coefficients = A_b
        self.coefficients = np.vstack([A_t, A_b])
        
        A_t_centered = A_t - A_t.mean(axis=0, keepdims=True)
        A_b_centered = A_b - A_b.mean(axis=0, keepdims=True)

        G = np.asarray(self.basis.gram_matrix(), dtype=np.float64)
        self.gram_matrix = G

        chol = np.linalg.cholesky(G)

        whitened_t = A_t_centered @ chol
        whitened_b = A_b_centered @ chol

        Sigma_t = compute_covariance(whitened_t)
        Sigma_b = compute_covariance(whitened_b)

        self.eigvals, eigvecs_w = generalized_eigen(
            Sigma_t,
            Sigma_b,
            method=method,
            mu=mu,
            n_components=self.n_components,
        )

        self.eigvecs_w = eigvecs_w
        self.eigvecs_coef = scipy.linalg.solve_triangular(
            chol.T,
            self.eigvecs_w,
            lower=False,
        )
        self.eigvecs = self.eigvecs_coef

        for idx in range(self.eigvecs_coef.shape[1]):
            vec = self.eigvecs_coef[:, idx]
            max_idx = np.argmax(np.abs(vec))
            if vec[max_idx] < 0:
                self.eigvecs_coef[:, idx] *= -1

        self.eigenfunction_dict = {
            component + 1: skfda.representation.basis.FDataBasis(
                self.basis,
                self.eigvecs_coef[:, component],
            )
            for component in range(self.eigvecs_coef.shape[1])
        }

        if self.eigvecs_coef.shape[1] >= 1:
            self.eigfunction_fd1 = self.eigenfunction_dict[1]
        if self.eigvecs_coef.shape[1] >= 2:
            self.eigfunction_fd2 = self.eigenfunction_dict[2]

        self.target_proj = A_t_centered @ G @ self.eigvecs_coef
        self.background_proj = A_b_centered @ G @ self.eigvecs_coef
#         self.loadings = self.eigvecs * np.sqrt(np.abs(self.eigvals))

        self.eigenfunctions_fd = skfda.representation.basis.FDataBasis(
            self.basis,
            self.eigvecs_coef.T,
        )
        eigen_vals = np.asarray(self.eigenfunctions_fd(self.time_points_), dtype=np.float64)
        if eigen_vals.ndim == 3 and eigen_vals.shape[-1] == 1:
            eigen_vals = eigen_vals[..., 0]
        self.eigenfunction_values = eigen_vals

        return self
