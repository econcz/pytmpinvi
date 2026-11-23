# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [5.0.0] – 2026-11-23

Version 5.0.0 adapts TMPinvi to pytmpinv 5.0.0 and pyclsp 5.0.0, which enforce bounds on the cells exactly. The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

### Added

- `TMPinviResult.replaced`: boolean m × p mask of the cells of `x` set to `replace_value` in `tmpinv()`, copied from `TMPinvResult.replaced` (all *False* if the result passed to `TMPinviResult()` has no such attribute).
- `TMPinviResult.sweeps`: the number of block sweeps of the reduced model, copied from `TMPinvResult.sweeps` (*None* in the full model or if the result passed to `TMPinviResult()` has no such attribute).
- Documentation of `sweep_limit` (pytmpinv 5.0.0), the maximum number of block sweeps of the reduced model, passed through to `tmpinv()`.
- Callable `ibounds`: a function `ibounds(ival)`, called after `preestimation()`, whose return value is used as the bounds (*None* leaves the cells unbounded).
- Validation of `ival` before the estimation: inf raises a *TMPinviInputError* ("ival must not contain inf."), and so does a shape different from (`len(b_row)`·`i`, `len(b_col)`·`j`) if `b_row` and `b_col` are given ("Dimensions of ival and the table (b_row, b_col, i, j) do not match.").
- Documentation of the pass-through of all options of pytmpinv 5.0.0 and pyclsp 5.0.0 (e.g. `zero_diagonal="exact"`, `slack_sign`, `r`, `Z`, `rcond`, `final`, `alpha`, `scale` and CVXPY solver arguments), of the `i` parameter of `summary()`, and of the attributes of the CLSP object passed to `postestimation()` (`status`, `rho`, `residual`, `fixed`, …).
- Test suite `tests/test_tmpinvi.py` (21 tests; references computed with numpy, pytmpinv's `tmpinv()` and pyclsp's CLSP, covering `ival` with missing entries, static, programmatic and callable `ibounds`, *np.nan* sides, exact bounds, the propagation of `replaced` to `data`, the hooks, the reduced model (block sweeps, `sweeps`, `sweep_limit`), `summary()`, the error cases and the validation of `alpha`; runs under pytest or as a script, against the installed package with `TMPINVI_TEST_INSTALLED=1`, with pytmpinv and pyclsp source trees from `TMPINV_SRC` and `CLSP_SRC`).
- CI: a PyPI workflow `pypi.yml` with a `test` job (Python 3.11 and 3.12), on which the release build depends, and a `tests.yml` workflow (push and pull request; Python 3.11, 3.12 and 3.13) that builds the wheel and runs the tests against the installed wheel. Both install pytmpinv ≥ 5.0.0 and pyclsp ≥ 5.0.0 from PyPI.
- Full MIT license text in `LICENSE` (previously only the title line).
- `CHANGELOG.md`; README sections on how TMPinvi works (what it adds, the order of operations, prior cell information, dynamic bounds, hooks, `update`, `replace_value` and `replaced`), six runnable examples with their output, an extended parameter reference, warnings and troubleshooting, limitations, and citation.

### Changed

- The bounds in `ibounds` (and `bounds`) are enforced exactly by CLSP through pytmpinv 5.0.0: by bounded-variable least squares in Step 1 and as constraints in Step 2, also in the reduced model. *None*, *np.nan* and ±*np.inf* denote an unbounded side, arrays of shape (m·p, 2) and tuples of pairs are accepted, and malformed bounds raise a *TMPinvInputError*.
- `replace_value` (pytmpinv 5.0.0) applies only to the cells of non-converged fits and to bound violations after the symmetrization, and *None* keeps the estimate; with `update=True`, the replaced cells are carried into `data` only where `ival` is missing.
- `alpha` is a single value in [0, 1] and *None* means α = 1 (pyclsp 5.0.0).
- Reduced model (pytmpinv 5.0.0): the full problem is solved by block-coordinate sweeps, each block fitted by CLSP with the other blocks fixed, so that the totals and the known cells are reproduced; `model` holds the block fits of the last sweep, and `postestimation(model_i, i)` is called for them. README Example 6 shows the sweeps instead of the overshooting independent blocks.
- Docstrings: the known values of `ival` are described as least-squares rows, the bounds as exactly enforced, and the summary line of `tmpinvi()` no longer refers to bound-constrained iterative refinement.
- README: the Quick Example uses a fixed range of years (reproducible output), passes the bounds as an (m·m, 2) array with *np.nan* sides, sets an exact zero diagonal, records ρ of every fit with a postestimation hook, and lists pandas and statsmodels as optional requirements of the example.
- Requires Python ≥ 3.11 and pytmpinv ≥ 5.0.0; classifiers for Python 3.11, 3.12 and 3.13; version 5.0.0 in `pyproject.toml` and `tmpinvi/__init__.py`.

### Fixed

- An `ival` with the number of cells of the table but a different shape (e.g. transposed) was flattened into a model matrix for the wrong cells; with `update=True`, the mismatch was detected only after the estimation.
- An `ival` with inf failed in `tmpinv()` with a message about `b_val`.
- Through pytmpinv 5.0.0: the reduced model did not reproduce the totals, since the blocks were fitted independently (README Example 6: row sums 1.6 to 2.0 times the row totals; now reproduced up to 2·10⁻¹⁴).
- Through pytmpinv 5.0.0 and pyclsp 5.0.0: with `zero_diagonal=True`, a known diagonal cell in `ival` silently removed its soft row x_kk = 0 (full and reduced model); both rows are now fitted.

### Removed

- Support for Python 3.10.
- Through pytmpinv 5.0.0 and pyclsp 5.0.0: the iterative bound-handling heuristic of `tmpinv()`, `alpha` lists and the NRMSE rule for `alpha=None`.

### Breaking

- pytmpinv ≥ 5.0.0 (and hence pyclsp ≥ 5.0.0) and Python ≥ 3.11 are required.
- Estimates with `ibounds` differ from 4.0.0: they are exact bounded least-squares estimates (Step 1) and, with `final=True`, the bounded table selected by α (Step 2), instead of the result of the heuristic; out-of-bound cells are no longer set to *np.nan* in `x` and `data`.
- `replace_value` has a new scope (non-converged fits and bound violations after the symmetrization), and `replace_value=None` keeps the estimate.
- `alpha` lists, tuples and arrays raise a *CLSPError*, and `alpha=None` means α = 1.
- `ival` with inf, or with a shape different from the table, raises a *TMPinviInputError* before the estimation.
- `S` must have `len(b_row)` + `len(b_col)` rows and uses signed slacks by default, and the reduced model is estimated by block sweeps (different estimates, no block slacks, `model` from the last sweep) and rejects grouped totals (pytmpinv 5.0.0).

## [4.0.0] – 2026-06-06

- Initial commit.
