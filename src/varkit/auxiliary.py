"""Auxiliary functions ported from MATLAB's VAR-Toolbox Auxiliary folder."""

import numpy as np
from scipy import stats, special

def ols(y: np.ndarray, x: np.ndarray, add_constant: bool = True) -> dict:
    """Least-squares regression.
    
    Args:
        y: Dependent variable vector (nobs x 1)
        x: Independent variables matrix (nobs x nvar)
        add_constant: Whether to add a constant term (default: True to match MATLAB behavior)
    
    Returns:
        Dictionary with regression results:
            - meth: 'ols'
            - beta: bhat (nvar x 1)
            - tstat: t-stats (nvar x 1)
            - bstd: std deviations for bhat (nvar x 1)
            - tprob: t-probabilities (nvar x 1)
            - yhat: yhat (nobs x 1)
            - resid: residuals (nobs x 1)
            - sige: e'*e/(n-k) scalar
            - rsqr: rsquared scalar
            - rbar: rbar-squared scalar
            - dw: Durbin-Watson Statistic
            - nobs: nobs
            - nvar: nvars
            - y: y data vector (nobs x 1)
            - bint: (nvar x 2) vector with 95% confidence intervals on beta
    """
    # Make sure y is a column vector
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    # Add constant term if requested (default behavior to match MATLAB)
    if add_constant:
        x = np.column_stack([np.ones(x.shape[0]), x])
    
    nobs, nvar = x.shape
    nobs2 = y.shape[0]
    
    if nobs != nobs2:
        raise ValueError('x and y must have same # obs in ols')
    
    results = {'meth': 'ols', 'y': y, 'nobs': nobs, 'nvar': nvar}
    
    if nobs < 10000:
        q, r = np.linalg.qr(x)
        xpxi = np.linalg.solve(r.T @ r, np.eye(nvar))
    else:  # use Cholesky for very large problems
        xpxi = np.linalg.solve(x.T @ x, np.eye(nvar))
    
    results['beta'] = xpxi @ (x.T @ y)
    results['yhat'] = x @ results['beta']
    results['resid'] = y - results['yhat']
    sigu = results['resid'].T @ results['resid']
    results['sige'] = sigu / (nobs - nvar)
    tmp = results['sige'] * np.diag(xpxi)
    sigb = np.sqrt(tmp)
    results['bstd'] = sigb
    tcrit = -stats.t.ppf(0.025, nobs)  # Equivalent to tdis_inv(.025,nobs)
    results['bint'] = np.column_stack([
        results['beta'] - tcrit * sigb,
        results['beta'] + tcrit * sigb
    ])
    results['tstat'] = results['beta'] / np.sqrt(tmp)
    
    # Calculate t-probabilities using tdis_prb
    results['tprob'] = tdis_prb(results['tstat'], nobs - nvar)
    
    ym = y - np.mean(y)
    rsqr1 = sigu
    rsqr2 = ym.T @ ym
    results['rsqr'] = 1.0 - rsqr1 / rsqr2  # r-squared
    rsqr1 = rsqr1 / (nobs - nvar)
    rsqr2 = rsqr2 / (nobs - 1.0)
    
    if rsqr2 != 0:
        results['rbar'] = 1 - (rsqr1 / rsqr2)  # rbar-squared
    else:
        results['rbar'] = results['rsqr']
    
    ediff = results['resid'][1:] - results['resid'][:-1]
    results['dw'] = (ediff.T @ ediff) / sigu  # durbin-watson
    
    return results

def tdis_inv(p: float, a: int) -> float:
    """Returns the inverse (quantile) at x of the t(n) distribution.
    
    Args:
        p: Probability
        a: Degrees of freedom
    
    Returns:
        Inverse of t-distribution at p with a degrees of freedom
    """
    return stats.t.ppf(p, a)

def beta_inv(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Inverse of the cdf (quantile) of the beta(a,b) distribution.
    
    Args:
        p: Vector of probabilities
        a: Beta distribution parameter a (scalar)
        b: Beta distribution parameter b (scalar)
    
    Returns:
        x at each element of p for the beta(a,b) distribution
    """
    return stats.beta.ppf(p, a, b)

def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """PDF of the beta(a,b) distribution.
    
    Args:
        x: Vector of components
        a: Beta distribution parameter a (scalar)
        b: Beta distribution parameter b (scalar)
    
    Returns:
        PDF at each element of x of the beta(a,b) distribution
    """
    return stats.beta.pdf(x, a, b)

def lag(x: np.ndarray, k: int) -> np.ndarray:
    """Create matrix of lagged values.
    
    Args:
        x: Data matrix (nobs x nvar)
        k: Number of lags
    
    Returns:
        Matrix of lagged values
    """
    if k == 0:
        return x
    
    nobs, nvar = x.shape
    X = np.zeros((nobs - k, nvar))
    
    # Lagged values
    X = x[:-k, :]
    
    return X

def nanmean(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean ignoring NaNs.
    
    Args:
        x: Input array
        axis: Axis along which to compute mean (0=columns, 1=rows)
    
    Returns:
        Mean of non-NaN elements
    """
    return np.nanmean(x, axis=axis)

def nanstd(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard deviation ignoring NaNs.
    
    Args:
        x: Input array
        axis: Axis along which to compute std (0=columns, 1=rows)
    
    Returns:
        Standard deviation of non-NaN elements
    """
    return np.nanstd(x, axis=axis)

def nanvar(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Variance ignoring NaNs.
    
    Args:
        x: Input array
        axis: Axis along which to compute variance (0=columns, 1=rows)
    
    Returns:
        Variance of non-NaN elements
    """
    return np.nanvar(x, axis=axis)

def nansum(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Sum ignoring NaNs.
    
    Args:
        x: Input array
        axis: Axis along which to compute sum (0=columns, 1=rows)
    
    Returns:
        Sum of non-NaN elements
    """
    return np.nansum(x, axis=axis)

def tdis_prb(t: np.ndarray, n: int) -> np.ndarray:
    """Returns the two-tailed probability for t-distribution.
    
    Args:
        t: t-statistics
        n: Degrees of freedom
    
    Returns:
        Two-tailed probability
    """
    return 2 * (1 - stats.t.cdf(np.abs(t), n))

def trimr(x: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Trim rows from top and bottom of matrix.
    
    Args:
        x: Input matrix
        n1: Number of rows to trim from top
        n2: Number of rows to trim from bottom
    
    Returns:
        Trimmed matrix
    """
    if n1 + n2 >= x.shape[0]:
        raise ValueError('Attempting to trim too many rows')
    return x[n1:x.shape[0]-n2, :] 