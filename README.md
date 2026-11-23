# Interactive Tabular Matrix Problems via Pseudoinverse Estimation

**Interactive Tabular Matrix Problems via Pseudoinverse Estimation (TMPinvi)** provides an interactive wrapper for the `tmpinv()` function from the [pytmpinv](https://pypi.org/project/pytmpinv/ "Tabular Matrix Problems via Pseudoinverse Estimation") package, with options extending its functionality to pre- and post-estimation processing and streamlined incorporation of prior cell information. The Tabular Matrix Problems via Pseudoinverse Estimation (TMPinv) is a two-stage estimation method that reformulates structured table-based systems — such as allocation problems, transaction matrices, and input–output tables — as structured least-squares problems. Based on the [Convex Least Squares Programming (CLSP)](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") framework, TMPinv solves systems with row and column constraints, block structure, and optionally reduced dimensionality by (1) constructing a canonical constraint form and applying a pseudoinverse-based projection, followed by (2) a convex-programming refinement stage to improve fit, coherence, and regularization (e.g., via Lasso, Ridge, or Elastic Net). Bounds on the cells (e.g., non-negativity, upper limits or fixed cells) are enforced exactly by CLSP; the totals and the known cell values are fitted in least squares (see [How TMPinvi Works](#how-tmpinvi-works) and [Limitations](#limitations)). All calculations are performed in numpy.float64 precision.

The method is described in Bolotov (2025), see [Bibliography](#bibliography). Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [Installation](#installation)
- [How TMPinvi Works](#how-tmpinvi-works): [What TMPinvi Adds](#what-tmpinvi-adds), [The Order of Operations](#the-order-of-operations), [Prior Cell Information](#prior-cell-information-ival), [Dynamic Bounds](#dynamic-bounds-ibounds), [Hooks](#pre--and-post-estimation-hooks), [`update`, `replace_value` and `replaced`](#update-replace_value-and-replaced)
- [Examples](#examples):
  [1. Quick Example](#1-quick-example-completing-a-panel-of-trade-matrices),
  [2. Prior Cell Information](#2-prior-cell-information-and-update),
  [3. Bounds](#3-static-programmatic-and-callable-bounds),
  [4. Hooks](#4-pre--and-post-estimation-hooks),
  [5. `replace_value`](#5-replace_value-replaced-and-update),
  [6. Reduced Models](#6-reduced-block-wise-models)
- [User Reference](#user-reference): [TMPinvi Parameters](#tmpinvi-parameters), [TMPinv Parameters](#tmpinv-parameters), [CLSP Parameters](#clsp-parameters), [Returns](#returns)
- [Warnings and Troubleshooting](#warnings-and-troubleshooting)
- [Limitations](#limitations)
- [What's New in 5.0.0](#whats-new-in-500)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements) · [Bibliography](#bibliography) · [License](#license)

## Installation

```bash
pip install pytmpinvi
```

The package requires Python ≥ 3.11 (tested on Python 3.11, 3.12 and 3.13) and depends on numpy and [pytmpinv](https://pypi.org/project/pytmpinv/ "Tabular Matrix Problems via Pseudoinverse Estimation") ≥ 5.0.0, which installs [pyclsp](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming") ≥ 5.0.0 together with scipy, cvxpy and the solvers ECOS, OSQP, SCS and CLARABEL. The [Quick Example](#1-quick-example-completing-a-panel-of-trade-matrices) additionally uses pandas and statsmodels, which are not dependencies of the package (`pip install pandas statsmodels`); the other examples use numpy only.

## How TMPinvi Works

### What TMPinvi Adds

`tmpinvi()` estimates the same model as `tmpinv()` (an allocation problem, AP (TM), of CLSP with row and column totals, optional known values, bounds, a zero diagonal, symmetry or reduced blocks) and adds four options for interactive and repeated use, e.g. in loops over years, regions or iterations of a data-completion workflow:

- **Prior cell information** (`ival`): an m × p matrix of known cell values with *np.nan* for the missing cells, instead of a model matrix `M` and a vector `b_val`.
- **Dynamic bounds** (`ibounds`): cell-value bounds that can be computed from the data, modified in place by `preestimation()`, or supplied as a function `ibounds(ival)`.
- **Hooks** (`preestimation`, `postestimation`): functions called before and after the estimation, e.g. to prepare the inputs or to record the diagnostics of every fit.
- **Updating** (`update`): the matrix `ival` with its missing cells filled by the estimate, returned as `result.data`.

All other arguments (`b_row`, `b_col`, `S`, `i`, `j`, `zero_diagonal`, `reduced`, `sweep_limit`, `symmetric`, `replace_value`, `tolerance`, `iteration_limit`, `cond_tolerance`) are passed to `tmpinv()`, and the options of CLSP (`r`, `Z`, `rcond`, `final`, `alpha`, `scale`, `slack_sign`, solver arguments) are passed on by `tmpinv()` to `CLSP.solve()`. For the estimator itself — the canonical form, the exact enforcement of the bounds by bounded-variable least squares, Step 2, the zero diagonal, symmetry and the reduced models — see the documentation of [pytmpinv](https://pypi.org/project/pytmpinv/ "Tabular Matrix Problems via Pseudoinverse Estimation") and [pyclsp](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming").

### The Order of Operations

A call of `tmpinvi()` performs the following steps:

1. `ival` is converted to a 2-D float64 array and checked (2-D, no inf, the shape of the table (m, p) = (`len(b_row)`·`i`, `len(b_col)`·`j`)); if `ival` is supplied, `M` and `b_val` in the keyword arguments are discarded, and if `ibounds` is supplied, `bounds` is discarded.
2. `preestimation(ival)` is called. It receives the array of step 1 (or *None*), so that changes made to it in place are used in the following steps.
3. The known cells of `ival` (row-major) become the model rows: `M` is the corresponding row subset of the identity matrix and `b_val` their values. If all cells are missing, no model rows are passed.
4. If `ibounds` is callable, it is called as `ibounds(ival)`, and its return value is used as the bounds.
5. `tmpinv(bounds=ibounds, M=M, b_val=b_val, **kwargs)` is solved: `tmpinv()` normalizes the bounds, builds the CLSP problem(s), `CLSP().solve(problem="ap", ...)`, and applies `symmetric` and `replace_value`.
6. `postestimation(model)` is called with the CLSP object of the full model, or `postestimation(model_i, i)` for each block fit of the last sweep of the reduced model.
7. `result.data` is built: `ival` with the missing cells filled from `result.x` (`update=True`), or `result.x`.

### Prior Cell Information (`ival`)

The known cells of `ival` are model rows of CLSP, i.e. *least-squares rows* like the totals: if they are consistent with the totals and the bounds, they are reproduced exactly (up to rounding, Example 2); otherwise, they are traded off against the totals (Example 3). To impose a known value exactly, fix the cell through its bounds, low = high, e.g. with a callable `ibounds` that returns `(v, v)` for the known cells (Example 3 (c)). In the reduced model, the known cells are mapped to the blocks (`M` is a row subset of the identity matrix, as required by `tmpinv()`) and fitted together with the totals by the block sweeps.

### Dynamic Bounds (`ibounds`)

`ibounds` accepts every form of `tmpinv(bounds=...)`: a single pair `(low, high)` for all cells, a list or tuple of one pair or of m·p pairs (row-major), or an array of shape (m·p, 2). *None*, *np.nan* and ±*np.inf* denote an unbounded side, so that the columns of a data frame with missing bounds can be passed directly (`np.column_stack([lb, ub])`, Example 1), and low = high fixes a cell. The bounds are enforced exactly by CLSP in both steps, also in the reduced model; malformed bounds raise a *TMPinvInputError*. "Dynamic" bounds can be supplied in three ways (Example 3):

- **static**: a pair, list or array computed before the call, e.g. from a regression (Example 1);
- **programmatic**: a mutable object (e.g. a list) that `preestimation(ival)` fills or modifies in place;
- **callable**: a function `ibounds(ival)` that returns the bounds; it is called after `preestimation()` and receives the same `ival` (*None* if no `ival` is supplied). A return value of *None* leaves the cells unbounded.

### Pre- and Post-Estimation Hooks

`preestimation(ival)` is called once before the estimation; `postestimation(model)` is called after it with the fitted CLSP object (`result.model`), or `postestimation(model_i, i)` for each block `i` of a reduced model (the block fits of the last sweep). The return values are ignored. The CLSP object exposes the outcome of the fit (pyclsp 5.0.0), e.g. `model.status` (`"step1"`, `"final"`, `"solver"`, …), `model.rho` (the smallest residual attainable under the bounds), `model.residual`, `model.fixed` (the components at a bound at the end of Step 1), `model.zhat0`, `model.rank` and the goodness-of-fit statistics, so that a postestimation hook can log or check every fit (Example 4). `model.x` is the CLSP estimate before the symmetrization and the replacement of `tmpinv()`.

### `update`, `replace_value` and `replaced`

With `update=True`, `result.data` is `ival` with its missing cells replaced by the corresponding cells of `result.x`; the known cells keep their values from `ival`, even if the estimate differs from them (soft rows) or is symmetrized. Without `ival` or with `update=False`, `result.data` is `result.x`.

Since pytmpinv 5.0.0, `replace_value` (*np.nan* by default) applies only to the cells of a (full or block) model whose bounded Step 1 did not converge (`model.status["step1"]` starting with *'not converged'*) and to the cells that violate their bounds after the symmetrization (`symmetric=True` with asymmetric bounds); *None* keeps the estimate. The replaced cells are reported in the boolean mask `result.replaced` (copied from the result of `tmpinv()`), and they are carried into `result.data` only where `ival` is missing (Example 5). A replaced cell in `result.data` therefore marks a missing cell that could not be estimated reliably.

## Examples

Each example below is self-contained; the output shown was produced by running it with pytmpinvi 5.0.0, pytmpinv 5.0.0 and pyclsp 5.0.0 (numpy 64-bit floats; the last digits of solver-dependent values may differ across platforms and solver versions).

### 1. Quick Example: Completing a Panel of Trade Matrices

```python
import numpy                   as np
import pandas                  as pd
import statsmodels.formula.api as smf
from   scipy.stats import norm
from   tmpinvi     import tmpinvi

# Reproducibility
rng = np.random.default_rng(123456789)

iso2 = ["CN", "DE", "JP", "NL", "US"]
T    = 10
year = np.arange(2016, 2016 + T)
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

    X_true[y] = X.copy()

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
    ub[ub < 0.0] = np.nan                                    # NaN: no upper bound

    df[f"_{nm}_lb"] = 0.0
    df[f"_{nm}_ub"] = ub

df_out = df.copy()

lb_cols = [f"_EX_{c}_lb" for c in iso2]
ub_cols = [f"_EX_{c}_ub" for c in iso2]

# Iterative completion/refinement
fits = {}

for step in range(1, 3):
    for y in year:
        idx = df_out["year"].eq(y)
        d   = df_out.loc[idx].copy()

        ival = d[ex_cols].to_numpy(dtype=np.float64)

        lb = d[lb_cols].to_numpy(dtype=np.float64).ravel(order="C")
        ub = d[ub_cols].to_numpy(dtype=np.float64).ravel(order="C")

        fit = tmpinvi(
            ival=ival,
            ibounds=np.column_stack([lb, ub]),               # (m * m, 2), NaN = unbounded
            b_row=d["EX"].to_numpy(dtype=np.float64),
            b_col=d["IM"].to_numpy(dtype=np.float64),
            zero_diagonal="exact",                           # x_kk = 0 via bounds
            alpha=1.0,
            update=True,
            postestimation=lambda model: fits.update(
                {(step, y): (model.status["step1"], model.rho)}
            ),
        )

        df_out.loc[idx, ex_cols] = fit.data

# Drop temporary bound columns
drop_cols = df_out.filter(regex=r"^_EX_.*_(lb|ub)$").columns
df_out = df_out.drop(columns=drop_cols)

# results
pd.set_option("display.width", 100)
print(df_out[df_out["year"].eq(year[-1])].round(1).to_string(index=False))
print("missing cells left:", int(df_out[ex_cols].isna().sum().sum()))
X_hat = {y: df_out.loc[df_out["year"].eq(y), ex_cols].to_numpy() for y in year}
print("max |row sums - EX|:",
      f"{max(np.abs(X_hat[y].sum(axis=1) - X_true[y].sum(axis=1)).max() for y in year):.2e}")
print("max |col sums - IM|:",
      f"{max(np.abs(X_hat[y].sum(axis=0) - X_true[y].sum(axis=0)).max() for y in year):.2e}")
print("min(X_hat):", min(X_hat[y].min() for y in year),
      " max |diag(X_hat)|:", max(np.abs(np.diag(X_hat[y])).max() for y in year))
print("step 1 status:", sorted({v[0] for v in fits.values()}))
for step in (1, 2):
    print(f"rho, pass {step}:", " ".join(f"{fits[step, y][1]:.1f}" for y in year))

x_true = np.concatenate([X_true[y].ravel() for y in year])
x_hat  = np.concatenate([X_hat[y].ravel()  for y in year])
ss_res = np.sum((x_true - x_hat) ** 2)
ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
print(f"R2_user_defined: {1 - ss_res / ss_tot:.4f}")
```

Output:

```text
 year iso2  EX_CN  EX_DE  EX_JP  EX_NL  EX_US     EX     IM
 2025   CN    0.0  236.2  794.2 1062.3  736.9 2829.6 3081.2
 2025   DE  935.6    0.0  619.9 1082.6 1479.1 4117.1 2602.1
 2025   JP 1055.4 1072.1    0.0  737.0  418.7 3283.3 2472.4
 2025   NL  347.4  156.7   22.2    0.0 1365.0 1891.4 3626.1
 2025   US  742.8 1137.1 1036.1  744.2    0.0 3660.2 3999.7
missing cells left: 0
max |row sums - EX|: 6.95e+02
max |col sums - IM|: 5.87e+02
min(X_hat): 0.0  max |diag(X_hat)|: 0.0
step 1 status: ['converged']
rho, pass 1: 379.4 180.9 0.0 569.6 0.0 645.3 341.0 0.0 339.1 0.0
rho, pass 2: 379.4 180.9 0.0 569.6 0.0 645.3 341.0 0.0 339.1 0.0
R2_user_defined: 0.9050
```

The panel contains ten 5 × 5 export matrices with a zero diagonal, of which about half of the cells are observed, together with the exports (`EX`, row totals) and imports (`IM`, column totals) of every country. The upper bounds of the cells are the upper limits of the 95% confidence intervals of the fitted means of a linear model by country and year (negative limits are set to *np.nan*, i.e. no upper bound), and the lower bounds are 0; they are passed as an (m·m, 2) array with *np.nan* sides. All missing cells are filled, and the bounds and the exact zero diagonal hold. In six of the ten years, however, the upper bounds are incompatible with the totals: the bounds are kept, and the totals are fitted in least squares with the smallest attainable residual ρ > 0 (recorded by the postestimation hook), so that the row and column sums deviate from `EX` and `IM` in these years. In the second pass, all cells are known: `update=True` leaves `data` unchanged, and the equal values of ρ confirm that the completed tables are the bounded least-squares fits of the first pass. R² refers to all cells, including the observed ones.

### 2. Prior Cell Information and `update`

```python
import numpy   as np
from   tmpinvi import tmpinvi

# AP (TM) 4×5 table with known row and column sums, 8 known cells (ival) and x ≥ 0

seed   = 9
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 4, 5                                                # matrix dimensions (m x p)
X_true = np.round(rng.uniform(0, 10, size=(m, p)), 1)        # non-negative X_true
X_true[rng.random((m, p)) < 0.4] = 0.0                       # empty cells
ival   = X_true.copy()                                       # prior cell information
ival[rng.random((m, p)) < 0.6] = np.nan                      # missing cells (np.nan)

# model
result = tmpinvi(
             ival=ival,                                      # known cells -> M, b_val
             ibounds=(0, None),                              # x ≥ 0, enforced exactly
             b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0),
             update=True                                     # fill the missing cells
         )
x_free = tmpinvi(ival=ival, b_row=X_true.sum(axis=1),        # the same model without bounds
                 b_col=X_true.sum(axis=0)).x

# results
print("true X:")
print(X_true)
print("ival:")
print(ival)
print("result.data (ival with the missing cells filled):")
print(np.round(result.data, 4))
known = ~np.isnan(ival)
print("known cells:", int(known.sum()), " model rows:",
      result.model.A.shape[0] - result.model.C_idx[0],
      " cells at 0 in Step 1:", int(result.model.fixed.sum()))
print("min(X_hat)             :", result.x.min())
print("min(X_hat), unbounded  :", np.round(x_free.min(), 4))
print("max |X_hat - X_true|    :", f"{np.max(np.abs(result.data - X_true)):.2e}")
print("max |known - ival|     :", f"{np.max(np.abs(result.x[known] - ival[known])):.2e}")
print("max |row sums - b_row| :", f"{np.max(np.abs(result.data.sum(axis=1) - X_true.sum(axis=1))):.2e}")
print("max |col sums - b_col| :", f"{np.max(np.abs(result.data.sum(axis=0) - X_true.sum(axis=0))):.2e}")
print("status:", result.model.status["step1"], "/", result.model.status["final"],
      f"rho = {result.model.rho:.2e}, replaced = {int(result.replaced.sum())}")
```

Output:

```text
true X:
[[0.  2.9 6.  7.8 7.2]
 [9.2 8.6 9.2 0.3 0. ]
 [0.  0.7 0.1 8.3 0. ]
 [0.  0.  7.1 0.  0. ]]
ival:
[[nan 2.9 6.  nan 7.2]
 [9.2 nan nan nan nan]
 [nan nan 0.1 8.3 0. ]
 [nan 0.  nan nan nan]]
result.data (ival with the missing cells filled):
[[0.  2.9 6.  7.8 7.2]
 [9.2 8.6 9.2 0.3 0. ]
 [0.  0.7 0.1 8.3 0. ]
 [0.  0.  7.1 0.  0. ]]
known cells: 8  model rows: 8  cells at 0 in Step 1: 4
min(X_hat)             : 0.0
min(X_hat), unbounded  : -2.3978
max |X_hat - X_true|    : 4.97e-14
max |known - ival|     : 1.70e-14
max |row sums - b_row| : 4.97e-14
max |col sums - b_col| : 3.55e-14
status: converged / optimal rho = 4.74e-14, replaced = 0
```

The 8 known cells of `ival` become 8 model rows (a row subset of the identity matrix). Without bounds, the minimum-norm estimate has negative cells (−2.3978); with `ibounds=(0, None)`, 4 cells are at the bound 0 in Step 1, and the totals and the known cells, which are consistent, are reproduced up to rounding (ρ ≈ 0). Here, the totals, the known cells and x ≥ 0 identify the table: `result.data`, i.e. `ival` with the 12 missing cells filled, equals the true table up to rounding.

### 3. Static, Programmatic and Callable Bounds

```python
import numpy   as np
from   tmpinvi import tmpinvi

# AP (TM) 4×4 table with known totals and a noisy prior for the first row (ival):
# static, programmatic and callable bounds (ibounds)

seed   = 3
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 4, 4                                                # matrix dimensions (m x p)
X_true = np.round(rng.uniform(1, 9, size=(m, p)), 1)         # positive X_true
ival   = np.full((m, p), np.nan)
ival[0] = np.round(X_true[0] * rng.uniform(0.7, 1.3, p), 1)  # noisy prior, first row
b_row  = X_true.sum(axis=1)                                  # row sums (length m)
b_col  = X_true.sum(axis=0)                                  # column sums (length p)
known  = ~np.isnan(ival)
print("first row: true =", X_true[0], "prior =", ival[0],
      "b_row[0] =", round(b_row[0], 1), "sum(prior) =", round(ival[0].sum(), 1))

def report(label, result):
    print(f"{label}: rho = {result.model.rho:.4f}, fixed = {int(result.model.fixed.sum())}")
    print("    first row:", np.round(result.x[0], 4),
          " row sums - b_row:", np.round(result.x.sum(axis=1) - b_row, 4))

# (a) static bounds: an (m * p, 2) array, np.nan = unbounded side
lohi   = np.column_stack([np.zeros(m * p), np.full(m * p, np.nan)])
res_a  = tmpinvi(ival=ival, ibounds=lohi, b_row=b_row, b_col=b_col)
report("(a) static, x ≥ 0             ", res_a)

# (b) programmatic bounds: a list filled in place by preestimation(ival)
bounds = []
def build_bounds(ival):
    bounds[:] = [(0, None) if np.isnan(v) else (0.99 * v, 1.01 * v)
                 for v in ival.ravel()]                      # prior ± 1%
res_b  = tmpinvi(ival=ival, ibounds=bounds, preestimation=build_bounds,
                 b_row=b_row, b_col=b_col)
report("(b) list, prior ± 1%          ", res_b)

# (c) callable bounds: ibounds(ival) is called after preestimation()
res_c  = tmpinvi(ival=ival,
                 ibounds=lambda v: [(0, None) if np.isnan(x) else (x, x)
                                    for x in v.ravel()],     # prior cells fixed
                 b_row=b_row, b_col=b_col)
report("(c) callable, prior cells fixed", res_c)
print("    prior cells reproduced exactly:", bool(np.all(res_c.x[known] == ival[known])))
```

Output:

```text
first row: true = [1.7 2.9 7.4 5.7] prior = [1.5 3.2 8.3 5. ] b_row[0] = 17.7 sum(prior) = 18.0
(a) static, x ≥ 0             : rho = 0.1359, fixed = 0
    first row: [1.4385 3.1385 8.2385 4.9385]  row sums - b_row: [ 0.0538 -0.0077 -0.0077 -0.0077]
(b) list, prior ± 1%          : rho = 0.1646, fixed = 4
    first row: [1.485 3.168 8.217 4.95 ]  row sums - b_row: [ 0.12   -0.0171 -0.0171 -0.0171]
(c) callable, prior cells fixed: rho = 0.3207, fixed = 4
    first row: [1.5 3.2 8.3 5. ]  row sums - b_row: [ 0.3    -0.0429 -0.0429 -0.0429]
    prior cells reproduced exactly: True
```

The prior values of the first row sum to 18.0, whereas the row total is 17.7, so that the known values and the totals are inconsistent. (a) With x ≥ 0 only (an array with *np.nan* upper bounds), the prior cells are least-squares rows like the totals and are traded off against them (ρ = 0.1359). (b) The list `bounds` is filled in place by `preestimation()` with a band of ±1% around the prior values: the four cells end at the lower ends of their bands (`fixed` = 4), and ρ = 0.1646. (c) The callable `ibounds(ival)` fixes the prior cells through low = high: they are reproduced exactly, and the misfit is carried by the totals alone (ρ = 0.3207, the first row sum exceeds its total by 0.3).

### 4. Pre- and Post-Estimation Hooks

```python
import numpy   as np
from   tmpinvi import tmpinvi

# Three 5×5 tables with incomplete prior information (ival), estimated in a loop: the
# hooks log the share of missing cells and the outcome of each CLSP fit

seed   = 42
rng    = np.random.default_rng(seed)
log    = []

def pre(ival):                                               # called as preestimation(ival)
    log.append({"missing": float(np.isnan(ival).mean())})

def post(model):                                             # called as postestimation(model)
    log[-1].update(step1=model.status["step1"], final=model.status["final"],
                   solver=model.status["solver"], rho=model.rho,
                   residual=model.residual, fixed=int(model.fixed.sum()))

for t in range(3):
    X_true = rng.uniform(0, 5, size=(5, 5)) * (rng.random((5, 5)) > 0.3)
    ival   = np.where(rng.random((5, 5)) < 0.4, X_true, np.nan)
    result = tmpinvi(ival=ival, ibounds=(0, None), preestimation=pre,
                     postestimation=post, update=True,
                     b_row=X_true.sum(axis=1),
                     b_col=X_true.sum(axis=0) * (1 + 0.02 * t))  # inconsistent for t > 0

for t, entry in enumerate(log):
    print(f"t = {t}: missing = {entry['missing']:.2f}, step1 = {entry['step1']}, "
          f"final = {entry['final']} ({entry['solver']}), rho = {entry['rho']:.4f}, "
          f"residual = {entry['residual']:.4f}, fixed = {entry['fixed']}")
result.summary(display=True)
```

Output:

```text
t = 0: missing = 0.64, step1 = converged, final = optimal (OSQP), rho = 0.0000, residual = 0.0000, fixed = 2
t = 1: missing = 0.60, step1 = converged, final = optimal (none (anchor feasible)), rho = 0.2153, residual = 0.2153, fixed = 0
t = 2: missing = 0.56, step1 = converged, final = optimal (OSQP), rho = 0.4739, residual = 0.4739, fixed = 1
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                 [0, inf]
  Slack sign (S ≥ 0):            False
  Column scaling:                False
  Fixed variables:                   1

Numerical Stability:
  kappaC:                       1.4142
  kappaB:                       1.0735
  kappaA:                       8.5231
  rank:                             20

Goodness of Fit:
  r2_partial:                   1.0000
  nrmse:                        0.0241
  nrmse_partial:             3.598e-15
  z_lower:        min= 0.000e+00  max=    3.6656  mean=    1.3049  sd=    1.3494
  z_upper:        min= 0.000e+00  max=    4.9216  mean=    1.7520  sd=    1.8118
  x_lower:        min= 0.000e+00  max=    3.6656  mean=    1.3049  sd=    1.3494
  x_upper:        min= 0.000e+00  max=    4.9216  mean=    1.7520  sd=    1.8118
  y_lower:        min=       nan  max=       nan  mean=       nan  sd=       nan
  y_upper:        min=       nan  max=       nan  mean=       nan  sd=       nan
```

The preestimation hook records the share of missing cells of each table, and the postestimation hook the outcome of each CLSP fit: the status of both steps and the solver of Step 2 (*'none (anchor feasible)'*: the unbounded anchor already satisfies the bounds, so no solver is needed), ρ, the residual and the number of cells at the bound 0 in Step 1. The first table is consistent (ρ ≈ 0); the column totals of the second and third tables are scaled by 1.02 and 1.04, so that ρ > 0. `summary()` prints the configuration and the diagnostics of the last fit: the known cells are reproduced (`r2_partial` = 1, `nrmse_partial` ≈ 0), and the misfit is carried by the totals.

### 5. `replace_value`, `replaced` and `update`

```python
import numpy   as np
from   tmpinvi import tmpinvi

# AP (TM) 3×3 symmetric table with an asymmetric cell-specific bound and a known cell:
# replace_value, replaced and update

X_true = np.array([[0.0, 2.0, 1.0],
                   [2.0, 0.0, 3.0],
                   [1.0, 3.0, 0.0]])
ival   = np.full((3, 3), np.nan)
ival[1, 0] = 2.0                                             # a known cell (x_21)
lohi   = [(0, None)] * 9                                     # x ≥ 0
lohi[1] = (0, 0.1)                                           # x_12 ≤ 0.1, but not x_21
for replace_value in (np.nan, None):
    result = tmpinvi(ival=ival, ibounds=lohi, update=True, symmetric=True,
                     replace_value=replace_value,
                     b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0))
    print(f"replace_value = {replace_value}: replaced cells =",
          np.argwhere(result.replaced).tolist())
    print("result.x:")
    print(np.round(result.x, 4))
    print("result.data:")
    print(np.round(result.data, 4))
print("CLSP estimate before the symmetrization (x_12 ≤ 0.1 holds):")
print(np.round(result.model.x, 4))
```

Output:

```text
replace_value = nan: replaced cells = [[0, 1], [1, 0]]
result.x:
[[0.8933    nan 1.0567]
 [   nan 2.2267 1.7233]
 [1.0567 1.7233 1.22  ]]
result.data:
[[0.8933    nan 1.0567]
 [2.     2.2267 1.7233]
 [1.0567 1.7233 1.22  ]]
replace_value = None: replaced cells = []
result.x:
[[0.8933 1.05   1.0567]
 [1.05   2.2267 1.7233]
 [1.0567 1.7233 1.22  ]]
result.data:
[[0.8933 1.05   1.0567]
 [2.     2.2267 1.7233]
 [1.0567 1.7233 1.22  ]]
CLSP estimate before the symmetrization (x_12 ≤ 0.1 holds):
[[0.8933 0.1    2.0067]
 [2.     2.2267 0.7733]
 [0.1067 2.6733 1.22  ]]
```

The bound x_12 ≤ 0.1 holds in the CLSP estimate (`result.model.x`), but not after the symmetrization, since the mirror cell x_21 is not bounded: the average 1.05 violates the bound. With the default `replace_value=np.nan`, both cells are replaced in `result.x` and marked in `result.replaced`. In `result.data`, only the missing cell x_12 carries the replacement, while the known cell x_21 keeps its value 2.0 from `ival`; with `replace_value=None`, the symmetric estimate is kept, and `result.data` is not symmetric, since the known cell keeps its prior value. Symmetric bounds (equal bounds for x_12 and x_21) avoid the conflict.

### 6. Reduced (Block-Wise) Models

```python
import numpy   as np
from   tmpinvi import tmpinvi

# AP (TM) 12×12 trade matrix with a zero diagonal and 30% of known cells (ival),
# estimated as reduced models of (7, 7), i.e. 2×2 blocks of 6×6 cells

seed   = 123456789
rng    = np.random.default_rng(seed)

# sample (dataset)
m, p   = 12, 12                                              # matrix dimensions (m x p)
X_true = np.abs(rng.normal(size=(m, p))) * (1 - np.eye(m))   # zero diagonal
ival   = np.where(rng.random((m, p)) < 0.3, X_true, np.nan)  # 30% known cells

# model
blocks = []
result = tmpinvi(
             ival=ival, ibounds=(0, None), update=True,
             postestimation=lambda model, i: blocks.append(  # called for each block i
                 (i, model.x.shape, model.status["step1"], model.residual)),
             b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0),
             zero_diagonal="exact",                          # x_kk = 0 via bounds
             reduced=(7, 7)                                  # reduced models of (7, 7)
         )

# results
for i, shape, step1, residual in blocks:                     # last sweep
    print(f"block {i}: shape = {shape}, step1 = {step1}, residual = {residual:.1e}")
known  = ~np.isnan(ival)
print("full:", result.full, " sweeps:", result.sweeps)
print("min(X_hat) =", result.x.min(),
      " max |diag(X_hat)| =", np.abs(np.diag(result.x)).max())
print("max |known - ival|:", f"{np.max(np.abs(result.x[known] - ival[known])):.2e}")
print("max |row sums - b_row|:",
      f"{np.max(np.abs(result.x.sum(axis=1) - X_true.sum(axis=1))):.1e}")
print("summary(i=0):", {k: result.summary(i=0)[k] for k in ("n_fixed", "rank")})

full   = tmpinvi(ival=ival, ibounds=(0, None), zero_diagonal="exact",
                 b_row=X_true.sum(axis=1), b_col=X_true.sum(axis=0))
print("full model: max |row sums - b_row| =",
      f"{np.max(np.abs(full.x.sum(axis=1) - X_true.sum(axis=1))):.1e}")
print("max |X_hat - X_full|:", f"{np.max(np.abs(result.x - full.x)):.4f}")
```

Output:

```text
block 0: shape = (6, 6), step1 = converged, residual = 3.1e-14
block 1: shape = (6, 6), step1 = converged, residual = 3.2e-14
block 2: shape = (6, 6), step1 = converged, residual = 2.4e-14
block 3: shape = (6, 6), step1 = converged, residual = 2.4e-14
full: False  sweeps: 3
min(X_hat) = 0.0  max |diag(X_hat)| = 0.0
max |known - ival|: 4.00e-15
max |row sums - b_row|: 1.8e-14
summary(i=0): {'n_fixed': 6, 'rank': 23}
full model: max |row sums - b_row| = 6.2e-14
max |X_hat - X_full|: 0.4828
```

With `reduced=(7, 7)`, the 12 × 12 table is estimated as 4 blocks of 6 × 6 cells by block-coordinate sweeps (pytmpinv 5.0.0): in each sweep, every block is fitted by CLSP with the other blocks fixed, and the sweeps stop when the largest change of a cell falls below the tolerance (here, after 3 sweeps, `result.sweeps`). The postestimation hook is called once, with each block model of the last sweep and its index. Every block fit converged with a zero residual, the bounds (x ≥ 0 and the exact zero diagonal) hold, and both the known cells of `ival` and the totals are reproduced; `summary(i=0)` returns the summary of the first block. The estimate is a least-squares fit of the full problem, but not the minimum-norm one selected by the full model, so that it differs from the full-model estimate (here, by up to 0.48 per cell). Use the reduced model when the full model is too large (see [Limitations](#limitations)).

## User Reference

For comprehensive information on the estimator's capabilities, advanced configuration options, and implementation details, please refer to the [pytmpinv module](https://pypi.org/project/pytmpinv/ "Tabular Matrix Problems via Pseudoinverse Estimation") and the [pyclsp module](https://pypi.org/project/pyclsp/ "Convex Least Squares Programming"), on which TMPinvi is based, and to the docstrings in `tmpinvi/tmpinvi.py`.

To ensure cross-platform reproducibility, all CLSP implementations use a modified condition number function based on singular values, with a relative cutoff equal to `cond_tolerance * the largest singular value`.

```python
from tmpinvi import tmpinvi
result = tmpinvi(ival, ibounds, preestimation, postestimation, update, **kwargs)
```

All arguments are best passed by keyword; positional arguments after `update` are passed to `tmpinv()`, whose first parameters are `bounds`, `replace_value`, `tolerance`, `iteration_limit` and `cond_tolerance`.

### TMPinvi Parameters

`ival` : *array_like* or *None*, default = *None*<br>
Prior information on known cell values, an m × p matrix with *np.nan* for the missing cells. If supplied and not entirely missing, `ival` is flattened (row-major) and used to construct `b_val` and the corresponding identity-subset model matrix `M` internally. Missing entries (*np.nan*) are ignored. If all entries of `ival` are *np.nan*, no prior information is used and `b_val` and `M` are not passed to `tmpinv()`. When `ival` is provided, it overrides any `b_val` or `M` arguments supplied through keyword arguments. The known values are least-squares rows, like the totals (see [Prior Cell Information](#prior-cell-information-ival)). `ival` must be 2-D, must not contain inf and, if `b_row` and `b_col` are given, must have the shape (`len(b_row)`·`i`, `len(b_col)`·`j`).

`ibounds` : *tuple*, *list*, *np.ndarray*, *callable*, or *None*, default = *None*<br>
Dynamic cell-value bounds passed to `tmpinv(bounds=...)`. The object supplied to `ibounds` may be created or modified programmatically (for example within `preestimation()`). If a single pair such as *(low, high)* is provided, it is applied uniformly to all cells. Alternatively, a list or tuple of `m` * `p` pairs (row-major) or an array of shape (`m` * `p`, 2) may be supplied to specify cell-specific bounds; *None*, *np.nan* and ±*np.inf* denote an unbounded side, and low = high fixes a cell. If callable, it is called as `ibounds(ival)` after `preestimation()`, and its return value is used as the bounds. The bounds are enforced exactly by CLSP. When `ibounds` is not *None*, it overrides any `bounds` argument supplied through keyword arguments. See [Dynamic Bounds](#dynamic-bounds-ibounds).

`preestimation` : *callable* or *None*, default = *None*<br>
A function executed prior to model estimation. If supplied, it is called as `preestimation(ival)` and may perform arbitrary preparatory steps, such as constructing dynamic bounds, modifying `ival` in place or modifying objects in the calling environment. The return value is ignored.

`postestimation` : *callable* or *None*, default = *None*<br>
A function executed after model estimation. For a full model, it is called as `postestimation(model)`. For reduced (block-wise) models, it is called as `postestimation(model_i, i)` for each block index `i` (the block fits of the last sweep). The model is the fitted CLSP object (with `x`, `status`, `rho`, `residual`, `fixed`, …), before the symmetrization and the replacement of `tmpinv()`. The return value is ignored.

`update` : *bool*, default = *False*<br>
If *True* and `ival` is supplied, missing entries (*np.nan*) in `ival` are replaced by the corresponding fitted values from `result.x`. The updated matrix is returned in the `result.data` component. If *False*, the data component contains the fitted solution matrix `result.x`. Cells set to `replace_value` in `result.x` are carried into `result.data` only where `ival` is missing.

### TMPinv Parameters

The following keyword arguments are passed to `tmpinv()`; see the pytmpinv documentation for the details.

`b_row` : *array_like* of shape *(m,)*<br>
Right-hand side vector of row totals (of row groups if `i` > 1). Please note that both `b_row` and `b_col` must be provided, with at least 2 finite entries each.

`b_col` : *array_like* of shape *(p,)*<br>
Right-hand side vector of column totals (of column groups if `j` > 1). Please note that both `b_row` and `b_col` must be provided, with at least 2 finite entries each.

`S` : *array_like* of shape *(m + p, m + p)*, optional<br>
A diagonal sign slack (surplus) matrix with entries in *{0, ±1}* (a zero-padded subset of ±I; zero columns may be omitted):<br>
-   *0* enforces equality (== `b_row` or `b_col`),<br>
-  *1* enforces a lower-than-or-equal (≤) condition,<br>
- *–1* enforces a greater-than-or-equal (≥) condition.

The first `m` diagonal entries correspond to row constraints, and the remaining `p` to column constraints; with grouped totals, `S` has `len(b_row)` + `len(b_col)` rows. The conditions are imposed through slacks y ≥ 0 (`slack_sign`, *True* by default when `S` is supplied): a total with S_kk = ±1 is free on its feasible side and remains a least-squares row on its infeasible side. Please note that, in the reduced model, `S` is ignored.

`M` : *array_like* of shape *(k, m * p)*, optional<br>
A model matrix with entries in *{0, 1}*, e.g. the selection of known cells, with the right-hand side `b_val` (least-squares rows). Overridden by `ival`; in the reduced model, `M` must consist of rows of an identity matrix (a cell may repeat).

`b_val` : *array_like* of shape *(k,)*<br>
Right-hand side vector of known cell values (finite). Overridden by `ival`.

`i` : *int*, default = *1*<br>
Number of rows in a row group: each entry of `b_row` is the sum of `i` consecutive rows, and the table has m = `len(b_row)` · `i` rows. Not supported in the reduced model.

`j` : *int*, default = *1*<br>
Number of columns in a column group: each entry of `b_col` is the sum of `j` consecutive columns, and the table has p = `len(b_col)` · `j` columns. Not supported in the reduced model.

`zero_diagonal` : *bool* or *'exact'*, default = *False*<br>
If *True*, adds the soft model rows x_kk = 0 for the diagonal cells (least-squares rows, traded off against the totals in an inconsistent system). If *'exact'*, sets the bounds of the diagonal cells to (0, 0), which makes them exactly zero; their bounds in `ibounds` must admit 0.

`reduced` : *tuple* of *(int, int)*, optional<br>
Dimensions of the reduced problem. If specified, the problem is estimated by block-coordinate sweeps over contiguous submatrices of the original table: each block is fitted by CLSP with the other blocks fixed (Example 6). For example, `reduced` = *(6, 6)* implies *5×5* data blocks (edge blocks may be smaller). Each dimension must be at least 3. The known cells of `ival` and the bounds are mapped to the blocks. See [Limitations](#limitations).

`sweep_limit` : *int*, default = *100*<br>
Maximum number of block sweeps in the reduced model (ignored in the full model). The sweeps stop when the largest change of a cell between two sweeps is below `tolerance` · max(1, max |x|); if this is not reached within `sweep_limit` sweeps, a *RuntimeWarning* is issued and the last table is returned. The number of sweeps performed is `TMPinviResult.sweeps`.

`symmetric` : *bool*, default = *False*<br>
If True, enforces symmetry of the estimated solution matrix as: x = 0.5 * (x + x.T), which requires m = p.
Applies to TMPinviResult.x only (after the sweeps in the reduced model). For TMPinviResult.model symmetry, add explicit symmetry constraints to M in a full-model solve instead of using this flag.

`bounds` : *sequence* of *(low, high)*, optional<br>
Bounds on cell values, in the forms of `ibounds` (except a callable). Overridden by `ibounds` if `ibounds` is not *None*.

`replace_value` : *float* or *None*, default = *np.nan*<br>
Replacement value for the cells of a full or block model whose bounded first step did not converge (`model.status["step1"]` starting with *'not converged'*), and for the cells that violate their bounds by more than `tolerance` after the symmetrization. If *None*, no cell is replaced. The replaced cells are marked in `TMPinviResult.replaced`. See [`update`, `replace_value` and `replaced`](#update-replace_value-and-replaced).

`tolerance` : *float*, default = *square root of machine epsilon*<br>
Tolerance passed to CLSP (the bound and KKT tolerances of the bounded first step, the convergence of the refinement for `r` > 1, the verification of the second step), the tolerance of the bound check of `replace_value`, and the convergence tolerance of the sweeps of the reduced model.

`iteration_limit` : *int*, default = *50*<br>
Maximum number of passes of the bounded first step (after which scipy's BVLS is used) and of iterations of the refinement loop (`r` > 1) in CLSP.

`slack_sign` : *bool* or *None*, default = *None*<br>
If *True*, the slacks of `S` must be ≥ 0. If *None*, *True* if `S` is supplied, *False* otherwise. Ignored in the reduced model.

`cond_tolerance` : *float* or *None*, default = *None*<br>
Singular-value cutoff for the custom condition number function. If *None*, the implementation uses an internal relative cutoff of `1e-14`.

### CLSP Parameters

The following keyword arguments are passed by `tmpinv()` to `CLSP.solve()` (of every block fit in the reduced model); see the pyclsp documentation for the details.

`r` : *int*, default = *1*<br>
Number of refinement iterations for the pseudoinverse-based estimator (used only with model rows, e.g. the known cells of `ival`). For `r` > 1, the misfit of the model rows is moved into residual slacks, which changes the estimator.

`Z` : *np.ndarray* or *None*<br>
A symmetric idempotent matrix (projector) defining the subspace for Bott–Duffin pseudoinversion. If *None*, the identity matrix is used, reducing the Bott–Duffin inverse to the Moore–Penrose case. With bounds, `Z` must be diagonal (0/1). Of use in the full model only.

`rcond` : *float*, *bool* or *None*, default = *None*<br>
Relative cutoff on the singular values of `A`·`Z`: the singular values σ ≤ `rcond`·σ_max are treated as zero in both steps (`model.rank`, `model.sv_discarded`). If *None* or *False*, the cutoff sqrt(max(`A`.shape)·eps) is applied; if *True*, a cutoff equal to `tolerance`.

`final` : *bool* or *None*, default = *None* (*True*)<br>
If *True*, Step 2 (the convex correction) is solved; if *False*, `x` is the Step 1 estimate.

`alpha` : *float* or *None*, default = *None*<br>
Regularization parameter (weight) of Step 2, a single value in [0, 1]:<br>
- `α = 0`: Lasso (L1 norm), solved with ECOS<br>
- `α = 1`: Tikhonov Regularization/Ridge (L2 norm), solved with OSQP (retried with CLARABEL if needed)<br>
- `0 < α < 1`: Elastic Net, solved with CLARABEL<br>
If *None*, α = 1. A *list*, *tuple* or *np.ndarray* of candidates, NaN, inf and values outside [0, 1] raise a *CLSPError* (the selection of α by NRMSE was removed in pyclsp 5.0.0; α can be chosen by cross-validation over the model rows with `result.model.cv_alpha()`). All values of α attain the same residual ρ; α only selects among the equally well-fitting tables, and without active bounds (or signed slacks), Step 2 returns the Step 1 estimate for every α.

`scale` : *bool* or *None*, default = *None* (*False*)<br>
If *True*, the columns of `A` are equilibrated in both steps (useful for totals of very different magnitudes); the minimum-norm selection is then made in the scaled variables.

`*args`, `**kwargs` : optional<br>
CVXPY arguments passed to the CVXPY solver of Step 2 (e.g., `solver="CLARABEL"` or solver tolerances).

### Returns

*TMPinviResult*

`TMPinviResult.full` : *bool*<br>
Indicates if this result comes from the full (non-reduced) model.

`TMPinviResult.model` : *CLSP* or *list* of *CLSP*<br>
A single CLSP object in the full model, or a list of CLSP objects for each reduced block in the reduced model (row-block by row-block; the fits of the last sweep). Each object exposes the CLSP results, e.g. `status`, `rho`, `residual`, `zhat`, `zhat0`, `rank`, `fixed`, the condition numbers, the goodness-of-fit statistics, and the methods `corr()`, `ttest()` and `cv_alpha()`.

`TMPinviResult.x` : *np.ndarray*<br>
Final estimated solution matrix of shape *(m, p)* (symmetrized if `symmetric=True`, with `replace_value` in the replaced cells).

`TMPinviResult.data` : *np.ndarray*<br>
Processed output matrix of shape *(m, p)*. If `update=True` and `ival` is supplied in `tmpinvi()`, `data` contains `ival` with missing entries replaced by fitted values from `x`. Otherwise, data contains the fitted solution matrix `x`.

`TMPinviResult.replaced` : *np.ndarray*<br>
Boolean mask of shape *(m, p)* of the cells of `x` set to `replace_value` in `tmpinv()` (all *False* if none).

`TMPinviResult.sweeps` : *int* or *None*<br>
Number of block sweeps in the reduced model (*None* in the full model), copied from the result of `tmpinv()`.

`TMPinviResult.summarize(i, display)`<br>
An alias of TMPinviResult.summary().

`TMPinviResult.summary(i, display)`<br>
Return or print a summary of the underlying CLSP result, where `i` : int, default = *None* is the index of a reduced-block model in TMPinviResult.model.

A fit is usable as intended if `model.status["step1"]` is *'converged'* or *'bvls fallback'* and `model.status["final"]` does not start with *'fallback'*.

## Warnings and Troubleshooting

Invalid inputs of `tmpinvi()` raise a *TMPinviInputError* (`tmpinvi.tmpinvi.TMPinviInputError`, with the method `as_dict()` and the attributes `message` and `code`); invalid arguments of `tmpinv()` (e.g. `ibounds`, `zero_diagonal`, the totals) raise a *TMPinvInputError* (`tmpinv.tmpinv.TMPinvInputError`), and invalid arguments of CLSP (e.g. `alpha`) a *CLSPError* (`clsp.errors.CLSPError`). A numerical step that does not reach its target issues a *RuntimeWarning* from pyclsp, and the result is still returned; the outcome is recorded in `model.status` (Example 4).

| Message | Cause and remedy |
|---|---|
| *TMPinviInputError*: ival must be a 2D array. / ival must not contain inf. | `ival` must be an m × p matrix with *np.nan* for the missing cells. |
| *TMPinviInputError*: Dimensions of ival and the table (b_row, b_col, i, j) do not match. / Dimensions of ival and result.x do not match. | `ival` must have the shape (`len(b_row)`·`i`, `len(b_col)`·`j`). |
| *TMPinviInputError*: preestimation must be callable. / postestimation must be callable. | A hook that is not a function. |
| *TMPinviInputError*: Reduced model: please supply the block index using i=#. / i must be in … | `summary()` of a reduced model without or with an invalid block index. |
| *TMPinvInputError*: Bounds length … does not match number of variables … / bounds must be None, a (low, high) pair, or a sequence of (low, high) pairs. / bounds must satisfy low ≤ high. | Malformed `ibounds` (or return value of a callable `ibounds`): the number of pairs must be 1 or `m`·`p`. |
| *TMPinvInputError*: zero_diagonal must be True, False or 'exact'. / zero_diagonal='exact' requires bounds that admit 0 on the diagonal. | Invalid `zero_diagonal`, or bounds of a diagonal cell that exclude 0. |
| *TMPinvInputError*: Both b_row and b_col must be provided. / Minimum length for b_row and b_col is 2. / b_row and b_col must not contain inf or NaN. | Missing, too short or non-finite totals. |
| *TMPinvInputError*: Each reduced block must be at least (3, 3) … / Grouped totals (i, j > 1) are not supported in the reduced model. / M must consist of rows of the identity matrix in the reduced model. / sweep_limit must be at least 1. | Unsupported reduced-model inputs. |
| *TMPinvInputError*: S must have … rows. / S must be a zero-padded subset of ±I. / symmetric=True requires a square matrix (m == p). | Invalid `S` or `symmetric=True` for a non-square table. |
| *CLSPError*: alpha must be a single value in [0, 1] … | A list, tuple or array of α, or an invalid α. |
| *UserWarning*: User-provided S is ignored in the reduced model. | `S` together with `reduced`. |
| *RuntimeWarning*: Reduced model: the block sweeps did not converge in … sweeps (…); increase sweep_limit. | The last table is returned; it may not reproduce the totals. Increase `sweep_limit` or use larger blocks. |
| *RuntimeWarning*: Bounded least-squares active set did not converge (…); using scipy BVLS | The result is a verified bounded least-squares solution (`status["step1"]` = *'bvls fallback'*); usually nothing to do. |
| *RuntimeWarning*: Bounded least-squares active set did not converge (…); the last feasible estimate is kept | Not a verified least-squares solution: the cells of this (block) model are set to `replace_value` (`result.replaced`). Try `scale=True`, a larger `iteration_limit` or a larger `rcond`. |
| *RuntimeWarning*: Step 2 failed (…); falling back to zhat | The Step 1 estimate (a bounded least-squares solution) is returned; try `scale=True` or another solver (`solver="CLARABEL"`). |

## Limitations

TMPinvi inherits the limitations of pytmpinv and pyclsp:

- **Soft totals and known values.** The totals, the known cells of `ival` and the rows of `zero_diagonal=True` are least-squares rows: in an inconsistent system, they are traded off against each other and are not met exactly (Examples 1, 3 and 4). Exact restrictions are available only as bounds (non-negativity, upper limits, fixed cells via low = high, `zero_diagonal="exact"`); a hard-constraint mode is not implemented in CLSP, and `S` with signed slacks makes a total an inequality only on its feasible side.
- **Reduced model.** The sweeps of a reduced model converge toward a least-squares fit of the full problem, but not necessarily the minimum-norm (Step 2) solution of the full model, so that the estimate differs from the full-model estimate when the least-squares solution is not unique (Example 6). The convergence is linear and can be slow (one CLSP fit per block and sweep); a sweep limit reached issues a *RuntimeWarning*. `M` (i.e. the known cells of `ival`) must select cells, `S` and grouped totals (`i`, `j` > 1) are not supported, and `result.model` holds the block fits of the last sweep only.
- **Symmetrization after the fit.** `symmetric=True` averages the estimate with its transpose after the fit: asymmetric bounds can then be violated (the cells are set to `replace_value`, Example 5), and known cells are preserved only if their mirror cells are known with the same values.
- **Step 2 needs active bounds.** Without bounds (and signed slacks), Step 2 returns the Step 1 estimate for every α.
- **Heuristic diagnostics.** The t-tests, the sensitivity bands and the correlogram of pyclsp (`model.ttest()`, `model.corr()`, the bands in `summary()`) are heuristic diagnostics without statistical coverage (see the pyclsp documentation).
- **Hooks are not sandboxed.** `preestimation()`, a callable `ibounds` and `postestimation()` run arbitrary user code; changes made by `preestimation()` to `ival` in place also change the caller's array if it is a float64 array.
- **Dense matrices.** All matrices are dense NumPy arrays, and the number of unknowns is m·p: the canonical matrix of a full model has (m + p + k) × m·p entries, and pyclsp computes its SVD and stores a (m·p) × (m·p) projector `Z`. The pyclsp documentation reports about 0.3 s and 20 MB for a 40 × 40 table and 4 s and 328 MB for an 80 × 80 table (`bounds=(0, None)`); larger tables require the reduced model or another method.

## What's New in 5.0.0

Version 5.0.0 adapts TMPinvi to pytmpinv 5.0.0 and pyclsp 5.0.0; the full list of changes is in [CHANGELOG.md](CHANGELOG.md).

- The bounds in `ibounds` are enforced exactly by CLSP (bounded-variable least squares in Step 1, constraints in Step 2) instead of the iterative heuristic of pytmpinv 4.0.0; *np.nan* and ±*np.inf* denote unbounded sides, arrays of shape (m·p, 2) and tuples of pairs are accepted, and malformed bounds raise a *TMPinvInputError*.
- `ibounds` can be a function `ibounds(ival)`, called after `preestimation()`.
- New attribute `TMPinviResult.replaced` (the cells set to `replace_value`); `replace_value` applies only to non-converged fits and to bound violations after the symmetrization, and *None* keeps the estimate.
- `ival` is checked for inf and for the shape of the table before the estimation.
- The reduced model of pytmpinv 5.0.0 solves the full problem by block-coordinate sweeps, so that the totals are reproduced; `sweep_limit` is passed through, and the number of sweeps is the new attribute `TMPinviResult.sweeps`.
- All options of pytmpinv 5.0.0 and pyclsp 5.0.0 are passed through, e.g. `zero_diagonal="exact"`, `slack_sign`, `r`, `Z`, `rcond`, `final`, `scale` and solver arguments; α is a single value in [0, 1] (α = 1 by default).
- A test suite, CI workflows, a CHANGELOG and an extended README; Python ≥ 3.11.

**Breaking changes:** pytmpinv ≥ 5.0.0 (and hence pyclsp ≥ 5.0.0) and Python ≥ 3.11 are required; the estimates with `ibounds` differ from 4.0.0 (exact bounded least squares instead of the heuristic), and out-of-bound cells are no longer set to *np.nan* in `x` and `data`; `replace_value` has a new meaning, and *None* now keeps the estimate; `alpha` lists and the NRMSE rule for `alpha=None` are removed (pyclsp); `ival` with inf or with a shape different from the table now raises a *TMPinviInputError* before the estimation; the reduced model is estimated by block sweeps (different estimates, `result.model` from the last sweep). See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use TMPinvi in your work, please cite the method:

Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

and the version of the package you used (`tmpinvi.__version__`):

pytmpinvi, version 5.0.0: Interactive Tabular Matrix Problems via Pseudoinverse Estimation [Python package]. [https://github.com/econcz/pytmpinvi](https://github.com/econcz/pytmpinvi)

```bibtex
@article{bolotov2025clsp,
  author  = {Bolotov, I.},
  title   = {{CLSP}: Linear Algebra Foundations of a Modular Two-Step Convex
             Optimization-Based Estimator for Ill-Posed Problems},
  journal = {Mathematics},
  year    = {2025},
  volume  = {13},
  number  = {21},
  pages   = {3476},
  doi     = {10.3390/math13213476}
}

@software{pytmpinvi,
  title   = {pytmpinvi: Interactive Tabular Matrix Problems via Pseudoinverse
             Estimation},
  version = {5.0.0},
  url     = {https://github.com/econcz/pytmpinvi}
}
```

## Acknowledgements

The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

## Bibliography
Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

## License

MIT License — see the [LICENSE](LICENSE) file.
