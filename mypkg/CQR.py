from rpy2 import robjects as robj
import numpy as np
r = robj.r
r["library"]("cfcausal");
r("""
CQR_fn <- function(X, Y, T, Xtest, nav=0, alpha=0.05, fyx_est="quantBoosting", estimand="unconditional", seed=0) {

    set.seed(seed)
    Y[T==nav] <- NA
    res <- conformalCf(X, Y, quantiles=c(alpha/2, 1-alpha/2), estimand=estimand, psfun=NULL, useCV=FALSE, outfun=fyx_est)
    CI = predict(res, Xtest, alpha=alpha);
    return(CI)
}
""")

def array2d2Robj(mat):
    """
    Converts a 2D numpy array to an R matrix object.

    Args:
        mat (numpy.ndarray): A 2D numpy array.

    Returns:
        r.matrix: An R matrix object.
    """
    mat_vec = mat.reshape(-1)
    mat_vecR = robj.FloatVector(mat_vec)
    matR = robj.r.matrix(mat_vecR, nrow=mat.shape[0], ncol=mat.shape[1], byrow=True)
    return matR

def get_CQR_CIs(X, Y, T, Xtest, nav=0, alpha=0.05, fyx_est="quantBoosting", estimand="unconditional", seed=0):
    """
    Calculates the conditional quantile regression confidence intervals.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        The input features.
    - Y: array-like, shape (n_samples,)
        The target variable.
    - T: array-like, shape (n_samples,)
        The treatment variable.
    - Xtest: array-like, shape (n_test_samples, n_features)
        The test input features.
    - nav: int, optional (default=0), 0 or 1
        make 0 or 1 as NA
    - alpha: float, optional (default=0.05)
        The significance level for calculating the confidence intervals.
    - fyx_est: str, optional (default="quantBoosting")
        The method for estimating the conditional quantile function.
    - seed: int, optional (default=0)
        The random seed for reproducibility.

    Returns:
    - CIs: array-like, shape (n_test_samples, 2)
        The lower and upper bounds of the confidence intervals for each test sample.
    """
    YR = robj.FloatVector(Y);
    TR = robj.FloatVector(T);
    XR = array2d2Robj(X);
    XtestR = array2d2Robj(Xtest);
    CIs = r["CQR_fn"](XR, YR, TR, XtestR, nav=nav, alpha=alpha, fyx_est=fyx_est, estimand=estimand, seed=seed) 
    CIs = np.array(CIs).T;
    return CIs
