"""
Regression and acceptance tests for pytmpinvi 5.0.0.

The references are computed with numpy, pytmpinv's tmpinv and pyclsp's CLSP
only. Runs under pytest or as a plain script: `python tests/test_tmpinvi.py`.
With the environment variable TMPINVI_TEST_INSTALLED=1, the installed package
is tested instead of the repo copy; TMPINV_SRC=<path> and CLSP_SRC=<path>
prepend a pytmpinv and a pyclsp source tree to sys.path.
"""
import io
import os
import sys
import traceback
import warnings
from   contextlib     import redirect_stdout
from   pathlib        import Path
import numpy          as np

ROOT      = Path(__file__).resolve().parents[1]
INSTALLED = os.environ.get("TMPINVI_TEST_INSTALLED") == "1"
for name in ("CLSP_SRC", "TMPINV_SRC"):              # source trees
    if os.environ.get(name):
        sys.path.insert(0, os.environ[name])
if not INSTALLED:                                    # test the repo copy
    sys.path.insert(0, str(ROOT))
import clsp                                          # noqa: E402
import clsp.solver                                   # noqa: E402
import tmpinv                                        # noqa: E402
import tmpinvi                                       # noqa: E402
from   clsp            import CLSP                   # noqa: E402
from   clsp.errors     import CLSPError              # noqa: E402
from   tmpinv          import tmpinv as tmp          # noqa: E402
from   tmpinv.tmpinv   import TMPinvInputError       # noqa: E402
from   tmpinvi         import tmpinvi as tmpi        # noqa: E402
from   tmpinvi.tmpinvi import TMPinviInputError      # noqa: E402
from   tmpinvi.tmpinvi import TMPinviResult          # noqa: E402

TOL      = 1e-8                                      # numerical tolerance
X_AP     = np.array([[.50, .25, .25],                # known AP case
                     [.25, 4.75, 5.0],
                     [.25, 5.0, 4.75]])
B_ROW    = X_AP.sum(1)                               # [1, 10, 10]
B_COL    = X_AP.sum(0)                               # [1, 10, 10]
X_AP_NN  = np.array([[0.0, .50, .50],                # min-norm, x ≥ 0
                     [.50, 4.75, 4.75],
                     [.50, 4.75, 4.75]])

# Ancillary functions (independent references)
def _table(seed, m, p, empty=0.3):
    """Random non-negative m×p table with empty cells."""
    rng                          = np.random.default_rng(seed)
    X                            = rng.uniform(0, 3, size=(m, p))
    X[rng.random((m, p)) < empty] = 0.0
    return X

def _ival(X, share, seed):
    """X with a random share of missing cells (np.nan)."""
    rng                          = np.random.default_rng(seed)
    ival                         = X.copy()
    ival[rng.random(X.shape) < share] = np.nan
    return ival

def _known(ival):
    """M and b_val of the known cells of ival (row-major)."""
    idx = np.flatnonzero(~np.isnan(ival.ravel()))
    return np.eye(ival.size)[idx], ival.ravel()[idx].reshape(-1, 1)

def _model_rows(model):
    """The model rows of the canonical A (on x)."""
    return model.A[model.C_idx[0]:, :model.C_idx[1]]

def _close(a, b, tol=TOL):
    return np.allclose(np.ravel(a), np.ravel(b), rtol=0.0, atol=tol,
                       equal_nan=True)

def _raises(func, error=TMPinviInputError, text=""):
    try:
        func()
    except error as e:
        assert text in str(e), str(e)
        return
    raise AssertionError(f"no {error.__name__} raised")

def _converged(model):
    return (model.status["step1"] in {"converged", "bvls fallback"} and
            not model.status["final"].startswith("fallback"))

def _patched(obj, name, value, func):
    """func() with obj.name temporarily replaced by value."""
    orig = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return func()
    finally:
        setattr(obj, name, orig)

def _step1_fails(fail):
    """A wrapper of clsp.solver._bvls reporting non-convergence if fail()."""
    bvls = clsp.solver._bvls
    def patched(*args, **kwargs):
        out = list(bvls(*args, **kwargs))
        if fail():
            out[3] = "not converged: test"
        return tuple(out)
    return patched

# 0. Package under test
def test_imports_package_under_test():
    path = Path(tmpinvi.__file__).resolve().parent
    assert (path != ROOT / "tmpinvi") if INSTALLED else (path == ROOT /
                                                         "tmpinvi")
    assert tmpinvi.__version__ == "5.0.0"
    assert tmpinv.__version__.split(".")[0] == "5"
    assert clsp.__version__.split(".")[0]   == "5"

# 1. Prior cell information (ival)
def test_no_ival_equals_tmpinv():
    for bounds in (None, (0, None)):
        res = tmpi(b_row=B_ROW, b_col=B_COL, bounds=bounds)
        ref = tmp(b_row=B_ROW, b_col=B_COL, bounds=bounds)
        assert isinstance(res, TMPinviResult) and res.full
        assert isinstance(res.model, CLSP)
        assert _close(res.x, ref.x, 0.0) and _close(res.data, res.x, 0.0)
        assert res.replaced.shape == (3, 3) and not res.replaced.any()
    assert _close(tmpi(b_row=B_ROW, b_col=B_COL, bounds=(0, None)).x,
                  X_AP_NN)

def test_ival_missing_entries():
    X       = _table(1, 5, 6)
    ival    = _ival(X, 0.5, 2)
    M, bv   = _known(ival)
    kw      = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None))
    ref     = tmp(M=M, b_val=bv, **kw)
    res     = tmpi(ival=ival, **kw)
    known   = ~np.isnan(ival)
    assert _close(res.x, ref.x, 0.0) and _close(res.data, res.x, 0.0)
    assert np.array_equal(_model_rows(res.model), M) # identity subset
    assert _close(res.x[known], X[known], 1e-7)      # consistent: exact
    assert _close(res.x.sum(1), X.sum(1), 1e-7) and _converged(res.model)
    res     = tmpi(ival=ival, update=True, **kw)
    assert _close(res.data[known], ival[known], 0.0) # known: unchanged
    assert _close(res.data[~known], res.x[~known], 0.0)
    assert not np.isnan(res.data).any()
    assert np.isnan(ival[~known]).all()              # input not modified

def test_ival_all_missing_and_overrides():
    X       = _table(3, 4, 4)
    kw      = dict(b_row=X.sum(1), b_col=X.sum(0), bounds=(0, None))
    M, bv   = _known(_ival(X, 0.5, 4))
    ref     = tmp(**kw)
    res     = tmpi(ival=np.full((4, 4), np.nan), M=M, b_val=bv,
                   update=True, **kw)
    assert _close(res.x, ref.x, 0.0) and _model_rows(res.model).size == 0
    assert _close(res.data, res.x, 0.0)              # all filled
    ival    = _ival(X, 0.6, 5)
    M2, bv2 = _known(ival)
    res     = tmpi(ival=ival, M=M, b_val=bv, **kw)   # ival overrides M
    assert _close(res.x, tmp(M=M2, b_val=bv2, **kw).x, 0.0)
    res     = tmpi(M=M, b_val=bv, **kw)              # kwargs without ival
    assert _close(res.x, tmp(M=M, b_val=bv, **kw).x, 0.0)

def test_ival_inconsistent_known_values():
    X       = _table(6, 5, 5)
    ival    = _ival(X, 0.7, 7)
    known   = ~np.isnan(ival)
    ival[known] += 0.5                               # noisy known values
    res     = tmpi(ival=ival, b_row=X.sum(1), b_col=X.sum(0),
                   ibounds=(0, None), update=True)
    assert res.model.rho > 0.1                       # soft rows
    assert _converged(res.model)
    assert np.min(res.x) >= 0.0                      # exact bounds
    assert _close(res.data[known], ival[known], 0.0)

# 2. Dynamic bounds (ibounds)
def test_ibounds_static_forms():
    kw      = dict(b_row=B_ROW, b_col=B_COL)
    for ibounds in ((0, None), (0, np.nan), (0.0, np.inf), [(0, None)],
                    [(0, None)] * 9, tuple([(0, np.nan)] * 9),
                    np.array([[0, np.nan]] * 9), [[0.0, None]] * 9):
        res = tmpi(ibounds=ibounds, **kw)
        assert _close(res.x, X_AP_NN) and res.model.status["rho"] == "bvls"
    for ibounds in ((None, None), (np.nan, np.nan), (-np.inf, np.inf),
                    np.full((9, 2), np.nan)):
        res = tmpi(ibounds=ibounds, **kw)
        assert res.model.bounds is None and res.x[0, 0] < 0
    res     = tmpi(ibounds=(0, None),                # overrides
                   bounds=(None, 1), **kw)
    assert _close(res.x, X_AP_NN)
    res     = tmpi(bounds=(0, None), **kw)           # bounds without ibounds
    assert _close(res.x, X_AP_NN)

def test_ibounds_nan_sides_cell_specific():
    X       = _table(8, 4, 5)
    lb      = np.where(X.ravel() > 1.0, 0.5, np.nan) # NaN: no lower bound
    ub      = np.where(X.ravel() < 2.0, X.ravel() + 0.1, np.nan)
    lohi    = np.column_stack([lb, ub])
    ref     = tmp(b_row=X.sum(1), b_col=X.sum(0),
                  bounds=[(None if np.isnan(l) else l,
                           None if np.isnan(h) else h) for l, h in lohi])
    res     = tmpi(ibounds=lohi, b_row=X.sum(1), b_col=X.sum(0))
    assert _close(res.x, ref.x, 0.0) and _converged(res.model)
    x       = res.x.ravel()
    assert np.all(x[~np.isnan(lb)] >= lb[~np.isnan(lb)])
    assert np.all(x[~np.isnan(ub)] <= ub[~np.isnan(ub)])

def test_ibounds_exact():
    for seed in range(4):
        X       = _table(20 + seed, 5, 6)
        ival    = _ival(X, 0.6, 30 + seed)
        hi      = float(np.max(X))
        lohi    = [(0, hi)] * 30
        lohi[7] = (X.ravel()[7], X.ravel()[7])       # a fixed cell
        res     = tmpi(ival=ival, ibounds=lohi, b_row=X.sum(1),
                       b_col=X.sum(0), update=True)
        assert np.min(res.x) >= 0.0 and np.max(res.x) <= hi
        assert res.x.ravel()[7] == X.ravel()[7] and _converged(res.model)
        assert _close(res.x.sum(1), X.sum(1), 1e-7)  # feasible: totals
        assert np.min(res.data) >= 0.0 and not res.replaced.any()
    res = tmpi(ibounds=(0, 4.6),                     # infeasible
               b_row=B_ROW, b_col=B_COL)
    assert np.min(res.x) >= 0.0 and np.max(res.x) <= 4.6
    assert abs(res.model.rho - 0.4899) < 1e-4 and _converged(res.model)

def test_ibounds_programmatic():
    X       = _table(9, 4, 4)
    ival    = _ival(X, 0.5, 10)
    kw      = dict(b_row=X.sum(1), b_col=X.sum(0))
    lohi    = [(None, None)] * 16                    # modified in place
    def pre(v):
        lohi[:] = [(0, None) if np.isnan(c) else (c, c) for c in v.ravel()]
    res     = tmpi(ival=ival, ibounds=lohi, preestimation=pre, **kw)
    known   = ~np.isnan(ival)
    assert np.min(res.x) >= 0.0 and _converged(res.model)
    assert np.array_equal(res.x[known], ival[known]) # fixed by bounds
    calls   = []
    def dyn(v):
        calls.append(v)
        return [(0, None) if np.isnan(c) else (c, c) for c in v.ravel()]
    res2    = tmpi(ival=ival, ibounds=dyn, **kw)
    assert len(calls) == 1 and _close(calls[0], ival, 0.0)
    assert _close(res2.x, res.x, 0.0)
    order   = []
    tmpi(ival=ival, preestimation=lambda v: order.append("pre"),
         ibounds=lambda v: order.append("ibounds"), **kw)
    assert order == ["pre", "ibounds"]
    res     = tmpi(ibounds=lambda v: None,           # overrides
                   bounds=(0, None),
                   b_row=B_ROW, b_col=B_COL)
    assert res.model.bounds is None and res.x[0, 0] < 0
    res     = tmpi(ibounds=lambda v: (0, None) if v is None else None,
                   b_row=B_ROW, b_col=B_COL)         # ival=None is passed
    assert _close(res.x, X_AP_NN)

# 3. Pre- and post-estimation hooks
def test_preestimation_hook():
    X       = _table(11, 4, 5)
    ival    = _ival(X, 0.5, 12)
    ival[0, 0] = np.nan
    seen    = []
    def pre(v):
        seen.append(v.copy())
        v[0, 0] = X[0, 0]                            # adds a known cell
    res     = tmpi(ival=ival.copy(), preestimation=pre, b_row=X.sum(1),
                   b_col=X.sum(0), bounds=(0, None), update=True)
    assert len(seen) == 1 and _close(seen[0], ival, 0.0)
    assert _model_rows(res.model).shape[0] == int((~np.isnan(ival)).sum()) + 1
    assert res.data[0, 0] == X[0, 0]
    seen    = []
    tmpi(preestimation=seen.append, b_row=B_ROW, b_col=B_COL)
    assert seen == [None]
    _raises(lambda: tmpi(preestimation=1, b_row=B_ROW, b_col=B_COL),
            text="preestimation must be callable")

def test_postestimation_hook():
    seen    = []
    def post(model, *i):
        seen.append((model, i, model.status["step1"], model.rho,
                     model.residual, model.fixed.copy()))
    res     = tmpi(b_row=B_ROW, b_col=B_COL, ibounds=(0, None),
                   postestimation=post)
    assert len(seen) == 1 and seen[0][0] is res.model and seen[0][1] == ()
    assert seen[0][2] == "converged" and seen[0][3] < TOL
    assert seen[0][4] < TOL and list(np.flatnonzero(seen[0][5])) == [0]
    X       = _table(13, 8, 8)
    seen    = []
    res     = tmpi(b_row=X.sum(1), b_col=X.sum(0), ibounds=(0, None),
                   reduced=(5, 5), postestimation=post)
    assert [s[1] for s in seen] == [(k,) for k in range(len(res.model))]
    assert all(s[0] is mod for s, mod in zip(seen, res.model))
    _raises(lambda: tmpi(postestimation="x", b_row=B_ROW, b_col=B_COL),
            text="postestimation must be callable")

# 4. replace_value and replaced
def test_replaced_symmetric_asymmetric_bounds():
    X       = np.array([[0.0, 2.0, 1.0], [2.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    lohi    = [(0, None)] * 9
    lohi[1] = (0, 0.1)                               # x_12 ≤ 0.1 only
    ival    = np.full((3, 3), np.nan)
    ival[1, 0] = 2.0                                 # a known mirror cell
    kw      = dict(b_row=X.sum(1), b_col=X.sum(0), ibounds=lohi,
                   symmetric=True)
    mask    = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    res     = tmpi(ival=ival, update=True, **kw)
    ref     = tmp(M=np.eye(9)[[3]], b_val=[2.0], b_row=X.sum(1),
                  b_col=X.sum(0), bounds=lohi, symmetric=True)
    assert np.array_equal(res.replaced, mask)
    assert np.array_equal(res.replaced, ref.replaced)
    assert _close(res.x, ref.x, 0.0) and np.isnan(res.x[mask]).all()
    assert np.isnan(res.data[0, 1]) and res.data[1, 0] == 2.0
    assert not np.isnan(res.data[~mask]).any()
    res     = tmpi(ival=ival, update=True, replace_value=None, **kw)
    assert not res.replaced.any() and not np.isnan(res.data).any()
    assert res.x[0, 1] > 0.1 and _close(res.x, res.x.T, 0.0)
    res     = tmpi(ival=ival, update=True, replace_value=-1.0, **kw)
    assert res.x[0, 1] == -1.0 and res.data[0, 1] == -1.0

def test_replaced_not_converged():
    X       = _table(14, 8, 8)
    ival    = _ival(X, 0.5, 15)
    kw      = dict(ival=ival, b_row=X.sum(1), b_col=X.sum(0),
                   ibounds=(0, None), update=True)
    fail    = _step1_fails(lambda: True)
    res     = _patched(clsp.solver, "_bvls", fail, lambda: tmpi(**kw))
    known   = ~np.isnan(ival)
    assert res.model.status["step1"] == "not converged: test"
    assert res.replaced.all() and np.isnan(res.x).all()
    assert np.isnan(res.data[~known]).all()          # missing: replaced
    assert _close(res.data[known], ival[known], 0.0)
    assert np.all(np.isfinite(res.model.x))          # model kept intact
    res     = _patched(clsp.solver, "_bvls", fail,
                       lambda: tmpi(replace_value=None, **kw))
    assert not res.replaced.any() and np.min(res.x) >= 0.0
    calls   = []
    fail    = _step1_fails(lambda: calls.append(1) or len(calls) % 4 == 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _patched(clsp.solver, "_bvls", fail,
                       lambda: tmpi(reduced=(5, 5), **kw))
    mask    = np.zeros((8, 8), dtype=bool)
    mask[:4, 4:] = True                              # second block only
    assert len(calls) == 4 * res.sweeps              # one Step 1 per block
    assert all(issubclass(v.category, RuntimeWarning) and
               "sweep_limit" in str(v.message) for v in w)
    assert np.array_equal(res.replaced, mask)
    assert np.isnan(res.data[mask & ~known]).all()
    assert not np.isnan(res.data[~mask]).any()

def test_result_container():
    class Result:
        full  = True
        model = None
        x     = [[1.0, 2.0], [3.0, 4.0]]
    res = TMPinviResult(Result(), None)              # no replaced attribute
    assert res.x.dtype == np.float64 and res.replaced.dtype == bool
    assert res.replaced.shape == (2, 2) and not res.replaced.any()
    Result.replaced = np.array([[True, False], [False, False]])
    res = TMPinviResult(Result(), None)
    assert res.replaced[0, 0] and res.replaced.sum() == 1

# 5. Reduced model and summary()
def test_reduced_model_passthrough():
    X       = _table(16, 10, 10) * (1 - np.eye(10))
    ival    = _ival(X, 0.7, 17)
    M, bv   = _known(ival)
    kw      = dict(b_row=X.sum(1), b_col=X.sum(0), reduced=(6, 6))
    ref     = tmp(M=M, b_val=bv, bounds=(0, None), zero_diagonal="exact",
                  **kw)
    res     = tmpi(ival=ival, ibounds=(0, None), zero_diagonal="exact",
                   update=True, **kw)
    known   = ~np.isnan(ival)
    assert not res.full and len(res.model) == 4
    assert _close(res.x, ref.x, 0.0) and res.sweeps == ref.sweeps
    assert np.min(res.x) >= 0.0 and np.all(np.diag(res.x) == 0.0)
    assert all(_converged(mod) for mod in res.model)
    assert _close(res.data[known], ival[known], 0.0)
    assert _close(res.data[~known], res.x[~known], 0.0)
    assert tmpi(b_row=B_ROW, b_col=B_COL).sweeps is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = tmpi(ival=ival, sweep_limit=1, **kw)
    assert res.sweeps == 1 and any("sweep_limit" in str(v.message)
                                   for v in w)
    _raises(lambda: tmpi(sweep_limit=0, **kw), TMPinvInputError,
            "sweep_limit")
    _raises(lambda: tmpi(b_row=[1.0, 2.0, 3.0], b_col=[2.0, 2.0, 2.0],
                         i=2, j=2, reduced=(4, 4)),
            TMPinvInputError, "Grouped totals")

def test_known_diagonal_cell():
    # a known x11 = 3 and the soft zero diagonal are both kept (pytmpinv)
    ival       = np.full((2, 2), np.nan)
    ival[0, 0] = 3.0
    for kw in ({}, {"reduced": (3, 3)}):
        res = tmpi(ival=ival, b_row=[2.0, 2.0], b_col=[2.0, 2.0],
                   zero_diagonal=True, **kw)
        assert _close(res.x, [[1.2, 1.1], [1.1, 0.6]], 1e-8), f"{kw}"

def test_reduced_repeated_rows():
    # repeated rows of M (ival = None) reach the block unchanged (pytmpinv)
    kw = dict(b_row=[2.0, 2.0], b_col=[2.0, 2.0], M=np.eye(4)[[0, 0]],
              b_val=[3.0, 0.0])
    for r in (1, 3):
        full = tmpi(r=r, **kw)
        res  = tmpi(r=r, reduced=(3, 3), **kw)
        assert _close(res.x, full.x, 1e-8), f"{r}"
        assert abs(res.model[0].rho - full.model.rho) < 1e-10

def test_summary():
    res     = tmpi(b_row=B_ROW, b_col=B_COL, ibounds=(0, None))
    out     = res.summary()
    assert isinstance(out, dict) and out == res.summarize()
    buf     = io.StringIO()
    with redirect_stdout(buf):
        res.summary(display=True)
    assert "Bounds (x)" in buf.getvalue()
    X       = _table(18, 8, 8)
    res     = tmpi(b_row=X.sum(1), b_col=X.sum(0), reduced=(5, 5))
    assert isinstance(res.summary(i=3), dict)
    assert isinstance(res.summarize(i=0), dict)
    _raises(lambda: res.summary(), text="block index")
    _raises(lambda: res.summary(i=4), text="i must be in")
    _raises(lambda: res.summary(i=-1), text="i must be in")

# 6. Options passed to tmpinv and CLSP, error cases
def test_kwargs_passthrough():
    X       = _table(19, 5, 5)
    ival    = _ival(X, 0.6, 20)
    M, bv   = _known(ival)
    kw      = dict(b_row=X.sum(1),
                   b_col=X.sum(0) * 1.05)            # inconsistent
    for opts in (dict(zero_diagonal="exact"), dict(zero_diagonal=True),
                 dict(final=False), dict(scale=True), dict(r=2),
                 dict(rcond=True), dict(alpha=0.0), dict(alpha=0.5),
                 dict(solver="CLARABEL"), dict(tolerance=1e-10),
                 dict(S=np.diag([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])),
                 dict(S=np.diag([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]),
                      slack_sign=False)):
        res = tmpi(ival=ival, ibounds=(0, None), **kw, **opts)
        ref = tmp(M=M, b_val=bv, bounds=(0, None), **kw, **opts)
        assert _close(res.x, ref.x, 1e-10), opts
        assert res.model.status == ref.model.status, opts
    res     = tmpi(ival=ival, ibounds=(0, None), zero_diagonal="exact", **kw)
    assert np.all(np.diag(res.x) == 0.0)
    res     = tmpi(ival=ival, ibounds=(0, None), final=False, **kw)
    assert res.model.status["final"] == "skipped"

def test_alpha_validation():
    kw      = dict(b_row=B_ROW, b_col=B_COL, ibounds=(0, None))
    for alpha in ([0.0, 1.0], (0.5,), np.array([0.5]), 2.0, -0.1, np.nan):
        _raises(lambda: tmpi(alpha=alpha, **kw), CLSPError, "alpha")
    res     = tmpi(alpha=None, **kw)
    assert res.model.alpha == 1.0
    assert _close(res.x, tmpi(alpha=1.0, **kw).x, 0.0)
    assert tmpi(alpha=0.25, **kw).model.alpha == 0.25

def test_error_cases():
    kw      = dict(b_row=B_ROW, b_col=B_COL)
    _raises(lambda: tmpi(ival=[1.0, 2.0], **kw), text="2D array")
    _raises(lambda: tmpi(ival=np.full((3, 3), np.inf), **kw), text="inf")
    _raises(lambda: tmpi(ival=np.ones((3, 4)), **kw), text="Dimensions")
    _raises(lambda: tmpi(ival=np.ones((2, 2)), b_row=[1.0, 1.0],
                         b_col=[2.0, 2.0], i=2, j=2), text="Dimensions")
    _raises(lambda: tmpi(ibounds=[(0, None)] * 8, **kw),
            TMPinvInputError, "length")
    _raises(lambda: tmpi(ibounds=(2, 1), **kw), TMPinvInputError,
            "low ≤ high")
    _raises(lambda: tmpi(ibounds="x", **kw), TMPinvInputError, "pair")
    _raises(lambda: tmpi(ibounds=lambda v: [(0, 1, 2)] * 9, **kw),
            TMPinvInputError, "pair")
    _raises(lambda: tmpi(zero_diagonal="yes", **kw), TMPinvInputError,
            "zero_diagonal")
    _raises(lambda: tmpi(b_row=B_ROW), TMPinvInputError, "Both")
    err     = TMPinviInputError("message", code=7)
    assert str(err) == "message [Code: 7]"
    assert err.as_dict() == {"error": "message", "code": 7}

# Plain-script runner
if __name__ == "__main__":
    tests  = [(k, f) for k, f in list(globals().items())
              if k.startswith("test_") and callable(f)]
    failed = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=-2)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
