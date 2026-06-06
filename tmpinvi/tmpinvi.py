import numpy                       as np
from   typing import Any, Callable

from   tmpinv import tmpinv

class TMPinviInputError(Exception):
    """
    Exception class for TMPinvi-related input errors.

    Represents internal failures in Interactive Tabular Matrix Problems via
    Pseudoinverse Estimation routines due to malformed or missing input.
    Supports structured messaging and optional diagnostic augmentation.

    Parameters
    ----------
    message : str, optional
        Description of the error. Defaults to a generic TMPinvi message.

    code : int or str, optional
        Optional error code or identifier for downstream handling.

    Attributes
    ----------
    message : str
        Human-readable error message.

    code : int or str
        Optional error code for custom handling or debugging.

    Usage
    -----
    raise TMPinviInputError("ival must be a 2D array", code=201)
    """
    def __init__(self, message: str = "An error occurred in TMPinvi",
                 code: int | str | None = None):
        self.message = message
        self.code    = code
        full_message = f"{message} (Code: {code})" if code is not None         \
                                                   else message
        super().__init__(full_message)

    def __str__(self) -> str:
        return self.message if self.code is None                               \
                            else f"{self.message} [Code: {self.code}]"

    def as_dict(self) -> dict:
        """
        Return the error as a dictionary for structured logging or JSON output.
        """
        return {"error": self.message, "code": self.code}

class TMPinviResult:
    """
    Result container for TMPinvi estimation.

    Attributes
    ----------
    full : bool
        Indicates if this result comes from the full (non-reduced) model.

    model : CLSP or list of CLSP
        A single CLSP object in the full model, or a list of CLSP objects
        for each reduced block in the reduced model.

    x : np.ndarray
        Final estimated solution matrix of shape (m, p).

    data : np.ndarray
        Processed output matrix of shape (m, p). If `update=True` and `ival` is
        supplied in `tmpinvi()`, `data` contains `ival` with missing entries
        replaced by fitted values from `x`. Otherwise, data contains the fitted
        solution matrix `x`.
    """
    def __init__(self, result, data: np.ndarray):
        self.full  = result.full
        self.model = result.model
        self.x     = np.asarray(result.x, dtype=np.float64)
        self.data  = data

    def summarize(self, i: int | None = None, display : bool = False):
        return self.summary(i=i, display=display)

    def summary(self, i: int | None = None, display: bool = False):
        """
        Return or print a summary for the TMPinvi estimator.

        Parameters
        ----------
        display: bool, default = False
            If True, prints the summary instead of returning a dictionary.
        """
        if not isinstance(self.model, list):
            return self.model.summarize(display=display)
        if i is None:
            raise TMPinviInputError("Reduced model: please supply the block " +
                                   "index using i=#.")
        idx = int(i)
        if idx < 0 or idx > len(self.model) - 1:
            raise TMPinviInputError(f"i must be in 0..{len(self.model)}-1 "   +
                                   f"for reduced model.")
        return self.model[idx].summarize(display=display)

def tmpinvi(
    ival:           Any                                       = None,
    ibounds:        Any                                       = None,
    preestimation:  Callable[[np.ndarray | None], Any] | None = None,
    postestimation: Callable[...,                 Any] | None = None,
    update:         bool                                      = False,
    *args, **kwargs
) -> TMPinviResult:
    """
    Solve an interactive tabular matrix estimation problem via Convex Least
    Squares Programming (CLSP) with bound-constrained iterative refinement.

    Parameters
    ----------
    ival : array_like or None, default = None
        Prior information on known cell values. If supplied and not entirely
        missing, `ival` is flattened and used to construct `b_val` and the
        corresponding identity-subset model matrix `M` internally. Missing
        entries (np.nan) are ignored. If all entries of `ival` are np.nan, no
        prior information is used and `b_val` and `M` are not passed to
        `tmpinv()`. When `ival` is provided, it overrides any `b_val` or `M`
        arguments supplied through keyword arguments.
    ibounds : tuple, list, or None, default = None
        Dynamic cell-value bounds passed to `tmpinv(bounds=...)`. The object
        supplied to `ibounds` may be created or modified programmatically
        (for example within `preestimation()`). If a single pair such as (low,
        high) is provided, it is applied uniformly to all cells. Alternatively,
        a list of pairs may be supplied to specify cell-specific bounds with
        others set to None or np.nan. When `ibounds` is not None, it overrides
        any `bounds` argument supplied through keyword arguments.
    preestimation : callable or None, default = None
        A function executed prior to model estimation. If supplied, it is
        called as `preestimation(ival)` and may perform arbitrary preparatory
        steps, such as constructing dynamic bounds or modifying objects in
        the calling environment. The return value is ignored.
    postestimation : callable or None, default = None
        A function executed after model estimation. For a full model, it is
        called as `postestimation(model)`. For reduced (block-wise) models,
        it is called as `postestimation(model_i, i)` for each block index
        `i`. The return value is ignored.
    update : bool, default = False
        If True and `ival` is supplied, missing entries (np.nan) in `ival`
        are replaced by the corresponding fitted values from `result.x`. The
        updated matrix is returned in the `result.data` component. If False,
        the data component contains the fitted solution matrix `result.x`.
    *args, **kwargs : additional arguments
        Passed directly to tmpinv().

    Returns
    -------
    TMPinviResult
        An object containing the fitted CLSP model(s), the solution matrix `x`,
        and the processed matrix `data`. If `update=True` and `ival` is
        supplied, `data` contains `ival` with missing entries replaced by
        fitted values from `x`. Otherwise, data contains the fitted solution
        matrix `x`.
    """
    # adjust and preprocess options
    if  ival             is not None:
        ival = np.asarray(ival, dtype=np.float64)
        if  ival.ndim  != 2:
            raise TMPinviInputError("ival must be a 2D array.")
        kwargs.pop("b_val",     None)
        kwargs.pop("M",         None)
    if  ibounds          is not None:
        kwargs.pop("bounds",    None)
    if  preestimation    is not None and not callable(preestimation):
        raise TMPinviInputError("preestimation must be callable.")
    if  postestimation   is not None and not callable(postestimation):
        raise TMPinviInputError("postestimation must be callable.")

    # run preestimation function
    if  preestimation    is not None:
        preestimation(ival)

    # construct prior-information block
    if  ival             is not None:
        if  not np.all(np.isnan(ival)):
            b_raw           = ival.ravel(order="C")
            idx             = ~np.isnan(b_raw)
            kwargs["b_val"] = b_raw[idx].reshape(-1, 1)
            kwargs["M"]     = np.eye(b_raw.size, dtype=np.float64)[idx]

    # perform estimation
    if  ibounds          is not None:
        kwargs["bounds"]    = ibounds
    result = tmpinv(*args, **kwargs)

    # run postestimation function
    if  postestimation   is not None:
        if  result.full:
            postestimation(result.model)
        else:
            for i, model_i in enumerate(result.model):
                postestimation(model_i, i)

    # generate, update, or replace data from result.x
    x = np.asarray(result.x, dtype=np.float64)
    if  update and ival  is not None:
        if  ival.shape != x.shape:
            raise TMPinviInputError(
                "Dimensions of ival and result.x do not match."
            )
        data           = ival.copy()
        missing        = np.isnan(data)
        data[missing]  = x[missing]
    else:
        data           = x

    return TMPinviResult(result, data)
