# Interactive Tabular Matrix Problems via Pseudoinverse Estimation

**Interactive Tabular Matrix Problems via Pseudoinverse Estimation (TMPinvi)** provides an interactive wrapper for the `tmpinv()` function from the *pytmpinv* package, with options extending its functionality to pre- and post-estimation processing and streamlined incorporation of prior cell information. The Tabular Matrix Problems via Pseudoinverse Estimation (TMPinv) is a two-stage estimation method that reformulates structured table-based systems — such as allocation problems, transaction matrices, and input–output tables — as structured least-squares problems. Based on the Convex Least Squares Programming (CLSP) framework, TMPinv solves systems with row and column constraints, block structure, and optionally reduced dimensionality by (1) constructing a canonical constraint form and applying a pseudoinverse-based projection, followed by (2) a convex-programming refinement stage to improve fit, coherence, and regularization (e.g., via Lasso, Ridge, or Elastic Net). All calculations are performed in numpy.float64 precision.

## Installation

```bash
pip install pytmpinvi
```

## Quick Example

```python
import numpy                   as np
import pandas                  as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from tmpinvi     import tmpinvi

# Reproducibility
rng = np.random.default_rng(123456789)

iso2 = ["CN", "DE", "JP", "NL", "US"]
T    = 10
year = np.arange(
    pd.Timestamp.today().year - T + 1,
    pd.Timestamp.today().year + 1
)
m    = len(iso2)

# Construct panel-like data frame
df = pd.MultiIndex.from_product(
    [year, iso2],
    names=["year", "iso2"]
).to_frame(index=False)

df = df.sort_values(["year", "iso2"]).reset_index(drop=True)

ex_cols = [f"EX_{c}" for c in iso2]

for nm in ex_cols:
    df[nm] = np.nan

df["EX"] = np.nan
df["IM"] = np.nan

X_true = {}

# Generate true transaction matrices and incomplete observations
for t, y in enumerate(year, start=1):
    scale = 1000.0 * (1.05 ** (t - 1))

    X = rng.uniform(0.0, scale, size=(m, m))
    np.fill_diagonal(X, 0.0)

    X_true[str(y)] = X.copy()

    rows = df["year"].eq(y)

    df.loc[rows, "EX"] = X.sum(axis=1)
    df.loc[rows, "IM"] = X.sum(axis=0)

    miss = rng.uniform(size=(m, m)) > 0.5

    X_obs = X.copy()
    X_obs[miss] = np.nan

    df.loc[rows, ex_cols] = X_obs

# Construct upper bounds using linear models
cv = norm.ppf(0.975)

for nm in ex_cols:
    fit = smf.ols(f"{nm} ~ year * C(iso2)", data=df).fit()

    pr = fit.get_prediction(df)
    sf = pr.summary_frame(alpha=0.05)

    ub = sf["mean"].to_numpy() + cv * sf["mean_se"].to_numpy()
    ub[ub < 0.0] = np.nan

    df[f"_{nm}_lb"] = 0.0
    df[f"_{nm}_ub"] = ub

def make_bounds(lb, ub):
    return [(a, b) for a, b in zip(lb, ub)]

df_out = df.copy()

lb_cols = [f"_EX_{c}_lb" for c in iso2]
ub_cols = [f"_EX_{c}_ub" for c in iso2]

# Iterative completion/refinement
for step in range(1, 3):
    for y in year:
        idx = df_out["year"].eq(y)
        d   = df_out.loc[idx].copy()

        ival = d[ex_cols].to_numpy(dtype=np.float64)

        lb = d[lb_cols].to_numpy(dtype=np.float64).ravel(order="C")
        ub = d[ub_cols].to_numpy(dtype=np.float64).ravel(order="C")

        fit = tmpinvi(
            ival=ival,
            ibounds=make_bounds(lb, ub),
            b_row=d["EX"].to_numpy(dtype=np.float64),
            b_col=d["IM"].to_numpy(dtype=np.float64),
            alpha=1.0,
            update=True,
        )

        df_out.loc[idx, ex_cols] = fit.data

# Drop temporary bound columns
drop_cols = df_out.filter(regex=r"^_EX_.*_(lb|ub)$").columns
df_out = df_out.drop(columns=drop_cols)

df_out
```

## User Reference

For comprehensive information on the estimator's capabilities, advanced configuration options, and implementation details, please refer to the [pytmpinv module](https://pypi.org/project/pytmpinv/ "Tabular Matrix Problems via Pseudoinverse Estimation"), on which TMPinvi is based.

To ensure cross-platform reproducibility, all CLSP implementations use a modified condition number function based on singular values, with a relative cutoff equal to `cond_tolerance * the largest singular value`.

**TMPinvi Parameters:**

`ival` : *array_like* or *None*, default = *None*<br>
Prior information on known cell values. If supplied and not entirely missing, `ival` is flattened and used to construct `b_val` and the corresponding identity-subset model matrix `M` internally. Missing entries (*np.nan*) are ignored. If all entries of `ival` are *np.nan*, no prior information is used and `b_val` and `M` are not passed to `tmpinv()`. When `ival` is provided, it overrides any `b_val` or `M` arguments supplied through keyword arguments.

`ibounds` : *tuple*, *list*, or *None*, default = *None*<br>
Dynamic cell-value bounds passed to `tmpinv(bounds=...)`. The object supplied to `ibounds` may be created or modified programmatically (for example within `preestimation()`). If a single pair such as *(low, high)* is provided, it is applied uniformly to all cells. Alternatively, a list of pairs may be supplied to specify cell-specific bounds with others set to None or *np.nan*. When `ibounds` is not *None*, it overrides any `bounds` argument supplied through keyword arguments.

`preestimation` : *callable* or *None*, default = *None*<br>
A function executed prior to model estimation. If supplied, it is called as `preestimation(ival)` and may perform arbitrary preparatory steps, such as constructing dynamic bounds or modifying objects in the calling environment. The return value is ignored.

`postestimation` : *callable* or *None*, default = *None*<br>
A function executed after model estimation. For a full model, it is called as `postestimation(model)`. For reduced (block-wise) models, it is called as `postestimation(model_i, i)` for each block index `i`. The return value is ignored.

`update` : *bool*, default = *False*<br>
If *True* and `ival` is supplied, missing entries (*np.nan*) in `ival` are replaced by the corresponding fitted values from `result.x`. The updated matrix is returned in the `result.data` component. If *False*, the data component contains the fitted solution matrix `result.x`.

**TMPinv Parameters:**

`S` : *array_like* of shape *(m + p, m + p)*, optional<br>
A diagonal sign slack (surplus) matrix with entries in *{0, ±1}*.<br>
-   *0* enforces equality (== `b_row` or `b_col`),<br>
-  *1* enforces a lower-than-or-equal (≤) condition,<br>
- *–1* enforces a greater-than-or-equal (≥) condition.

The first `m` diagonal entries correspond to row constraints, and the remaining `p` to column constraints. Please note that, in the reduced model, `S` is ignored: slack behavior is derived implicitly from block-wise marginal totals.

`M` : *array_like* of shape *(k, m * p)*, optional<br>
A model matrix with entries in *{0, 1}*. Each row defines a linear restriction on the flattened solution matrix. The corresponding right-hand side values must be provided in `b_val`. This block is used to encode known cell values. Please note that, in the reduced model, `M` must be a unique row subset of an identity matrix (i.e., diagonal-only). Arbitrary or non-diagonal model matrices cannot be mapped to reduced blocks, making the model infeasible.

`b_row` : *array_like* of shape *(m,)*<br>
Right-hand side vector of row totals. Please note that both `b_row` and `b_col` must be provided.

`b_col` : *array_like* of shape *(p,)*<br>
Right-hand side vector of column totals. Please note that both `b_row` and `b_col` must be provided.

`b_val` : *array_like* of shape *(k,)*<br>
Right-hand side vector of known cell values.

`i` : *int*, default = *1*<br>
Number of row groups.

`j` : *int*, default = *1*<br>
Number of column groups.

`zero_diagonal` : *bool*, default = *False*<br>
If *True*, enforces the zero diagonal.

`reduced` : *tuple* of *(int, int)*, optional<br>
Dimensions of the reduced problem. If specified, the problem is estimated as a set of reduced problems constructed from contiguous submatrices of the original table. For example, `reduced` = *(6, 6)* implies *5×5* data blocks with *1* slack row and *1* slack column each (edge blocks may be smaller).

`symmetric` : *bool*, default = *False*<br>
If True, enforces symmetry of the estimated solution matrix as: x = 0.5 * (x + x.T)
Applies to TMPinviResult.x only. For TMPinviResult.model symmetry, add explicit symmetry constraints to M in a full-model solve instead of using this flag.

`bounds` : *sequence* of *(low, high)*, optional<br>
Bounds on cell values. If a single tuple *(low, high)* is given, it is applied to all `m` * `p` cells. Example: *(0, None)*.

`replace_value` : *float* or *None*, default = *np.nan*<br>
Final replacement value for any cell in the solution matrix that violates the specified bounds by more than the given tolerance.

`tolerance` : *float*, default = *square root of machine epsilon*<br>
Convergence tolerance for bounds.

`iteration_limit` : *int*, default = *50*<br>
Maximum number of iterations allowed in the refinement loop.

**CLSP Parameters:**

`r` : *int*, default = *1*<br>
Number of refinement iterations for the pseudoinverse-based estimator.

`Z` : *np.ndarray* or *None*<br>
A symmetric idempotent matrix (projector) defining the subspace for Bott–Duffin pseudoinversion. If *None*, the identity matrix is used, reducing the Bott–Duffin inverse to the Moore–Penrose case.

`final` : *bool*, default = *True*<br>
If *True*, a convex programming problem is solved to refine `zhat`. The resulting solution `z` minimizes a weighted L1/L2 norm around `zhat` subject to `Az = b`.

`alpha` : *float*, *list[float]* or *None*, default = *None*<br>
    Regularization parameter (weight) in the final convex program:<br>
    - `α = 0`: Lasso (L1 norm)<br>
    - `α = 1`: Tikhonov Regularization/Ridge (L2 norm)<br>
    - `0 < α < 1`: Elastic Net<br>
    If a scalar float is provided, that value is used after clipping to [0, 1].<br>
    If a list/iterable of floats is provided, each candidate is evaluated via a full solve, and the α with the smallest NRMSE is selected.<br>
    If None, α is chosen, based on an error rule: α = min(1.0, NRMSE_{α = 0} / (NRMSE_{α = 0} + NRMSE_{α = 1} + tolerance))

`cond_tolerance` : *float* or *None*, default = *None*<br>
    Singular-value cutoff for the custom condition number function.<br>
    If *None*, the implementation uses an internal relative cutoff of `1e-14`.

`*args`, `**kwargs` : optional<br>
CVXPY arguments passed to the CVXPY solver.

**Returns:**
*TMPinviResult*

`TMPinviResult.full` : *bool*<br>
Indicates if this result comes from the full (non-reduced) model.

`TMPinviResult.model` : *CLSP* or *list* of *CLSP*<br>
A single CLSP object in the full model, or a list of CLSP objects for each reduced block in the reduced model.

`TMPinviResult.x` : *np.ndarray*<br>
Final estimated solution matrix of shape *(m, p)*.

`TMPinviResult.data` : *np.ndarray*<br>
Processed output matrix of shape *(m, p)*. If `update=True` and `ival` is supplied in `tmpinvi()`, `data` contains `ival` with missing entries replaced by fitted values from `x`. Otherwise, data contains the fitted solution matrix `x`.

`TMPinviResult.summarize(i, display)`<br>
An alias of TMPinviResult.summary().

`TMPinviResult.summary(i, display)`<br>
Return or print a summary of the underlying CLSP result, where `i` : int, default = *None* is the index of a reduced-block model in TMPinviResult.model.

## Bibliography
Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

## License

MIT License — see the [LICENSE](LICENSE) file.
