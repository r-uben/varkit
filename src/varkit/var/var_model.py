"""
Vector Autoregression (VAR) Model Implementation.

This module implements the main VAR model class, corresponding to the original VARmodel.m
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats


@dataclass
class VAROptions:
    """Class to hold VAR model options."""
    vnames: List[str] = field(default_factory=list)  # endogenous variables names
    vnames_ex: List[str] = field(default_factory=list)  # exogenous variables names
    snames: List[str] = field(default_factory=list)  # shocks names
    nsteps: int = 40  # number of steps for computation of IRFs and FEVDs
    impact: int = 0  # size of the shock for IRFs: 0=1stdev, 1=unit shock
    shut: int = 0  # forces the IRF of one variable to zero
    ident: str = 'short'  # identification method for IRFs
    recurs: str = 'wold'  # method for computation of recursive stuff
    ndraws: int = 1000  # number of draws for bootstrap or sign restrictions
    mult: int = 10  # multiple of draws to be printed at screen
    pctg: float = 95  # confidence level for bootstrap
    method: str = 'bs'  # methodology for error bands
    sr_hor: int = 1  # number of periods that sign restrictions are imposed on
    sr_rot: int = 500  # max number of rotations for finding sign restrictions
    sr_draw: int = 100000  # max number of total draws for finding sign restrictions
    sr_mod: int = 1  # model uncertainty for sign restrictions


def var_make_lags(data: Union[np.ndarray, pd.DataFrame], lag: int) -> np.ndarray:
    """Create matrix of lagged values, following VARmakelags.m.
    
    Args:
        data: Matrix containing the original data (nobs x nvar)
        lag: Lag order
    
    Returns:
        Matrix of lagged values
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    nobs = len(data)
    
    # Create the lagged matrix
    out = np.array([])
    for jj in range(lag):
        if out.size == 0:
            out = data[jj:nobs-lag+jj]
        else:
            out = np.hstack([data[jj:nobs-lag+jj], out])
    
    return out


def var_make_xy(data: Union[np.ndarray, pd.DataFrame], lags: int, const: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create matrices Y and X for VAR estimation, following VARmakexy.m.
    
    Args:
        data: Matrix containing the original data (nobs x nvar)
        lags: Lag order of the VAR
        const: Type of deterministic terms
               0: no constant, no trend
               1: constant, no trend
               2: constant, trend
               3: constant, trend^2
    
    Returns:
        tuple:
            - Y: VAR dependent variable
            - X: VAR independent variable
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    nobs = len(data)
    
    # Y matrix
    Y = data[lags:]
    
    # X-matrix
    X = np.array([])
    for jj in range(lags):
        if X.size == 0:
            X = data[jj:nobs-lags+jj]
        else:
            X = np.hstack([data[jj:nobs-lags+jj], X])
    
    # Add deterministic terms
    if const == 0:
        pass  # No constant, no trend
    elif const == 1:  # constant
        X = np.hstack([np.ones((nobs-lags, 1)), X])
    elif const == 2:  # time trend and constant
        trend = np.arange(1, nobs-lags+1).reshape(-1, 1)
        X = np.hstack([np.ones((nobs-lags, 1)), trend, X])
    elif const == 3:  # linear time trend, squared time trend, and constant
        trend = np.arange(1, nobs-lags+1).reshape(-1, 1)
        X = np.hstack([np.ones((nobs-lags, 1)), trend, trend**2, X])
    
    return Y, X


class VARModel:
    """Vector Autoregression (VAR) Model.
    
    This class implements VAR estimation with OLS, following Gertler and Karadi (2015).
    """
    
    def __init__(self, 
                 endo: pd.DataFrame,
                 nlag: int,
                 const: int = 1,
                 exog: Optional[pd.DataFrame] = None,
                 nlag_ex: int = 0):
        """Initialize VAR model.
        
        Args:
            endo: DataFrame of endogenous variables (nobs x nvar)
            nlag: Number of lags
            const: Type of deterministic terms
                  0: no constant
                  1: constant
                  2: constant and trend
                  3: constant and trend^2
            exog: Optional DataFrame of exogenous variables (nobs x nvar_ex)
            nlag_ex: Number of lags for exogenous variables
        """
        # Store inputs
        self.endo = endo
        self.nlag = nlag
        self.const = const
        self.exog = exog
        self.nlag_ex = nlag_ex
        
        # Get dimensions
        self.nobs, self.nvar = endo.shape
        self.nvar_ex = exog.shape[1] if exog is not None else 0
        
        # Validate inputs
        self._validate_inputs()
        
        # Create options
        self.options = VAROptions()
        
        # Compute effective sample size
        self.nobse = self.nobs - max(self.nlag, self.nlag_ex)
        
        # Compute number of coefficients
        self.ncoeff = self.nvar * self.nlag
        self.ncoeff_ex = self.nvar_ex * (self.nlag_ex + 1)
        self.ntotcoeff = self.ncoeff + self.ncoeff_ex + self.const
        
        # Initialize results dictionary with model attributes
        self.results = {
            'ENDO': self.endo.values,  # Store as numpy array
            'EXOG': self.exog.values if self.exog is not None else None,
            'nvar': self.nvar,
            'nvar_ex': self.nvar_ex,
            'nlag': self.nlag,
            'nlag_ex': self.nlag_ex,
            'const': self.const,
            'nobs': self.nobse,  # Use effective sample size
            'ncoeff': self.ncoeff,
            'ncoeff_ex': self.ncoeff_ex,
            'ntotcoeff': self.ntotcoeff
        }
        
        # Estimate VAR
        self._estimate()
    
    def _validate_inputs(self):
        """Validate input data."""
        if self.exog is not None:
            if len(self.exog) != len(self.endo):
                raise ValueError('Endogenous and exogenous variables must have same number of observations')
    
    def _estimate(self):
        """Estimate VAR model using OLS."""
        # Create Y and X matrices for endogenous variables
        Y, X = var_make_xy(self.endo, self.nlag, self.const)
        
        # Add exogenous variables if present
        if self.exog is not None and self.nvar_ex > 0:
            X_EX = var_make_lags(self.exog, self.nlag_ex)
            
            # Align matrices based on lag lengths
            if self.nlag == self.nlag_ex:
                X = np.hstack([X, X_EX])
            elif self.nlag > self.nlag_ex:
                diff = self.nlag - self.nlag_ex
                X_EX = X_EX[diff:]
                X = np.hstack([X, X_EX])
            else:  # nlag < nlag_ex
                diff = self.nlag_ex - self.nlag
                Y = Y[diff:]
                X = np.hstack([X[diff:], X_EX])
        
        # Convert back to DataFrames with proper indices and column names
        Y = pd.DataFrame(Y, index=self.endo.index[self.nlag:], columns=self.endo.columns)
        
        # Create column names for X
        X_cols = []
        if self.const >= 1:
            X_cols.append('const')
        if self.const >= 2:
            X_cols.append('trend')
        if self.const >= 3:
            X_cols.append('trend2')
        
        # Add endogenous variables column names
        for lag in range(1, self.nlag + 1):
            for col in self.endo.columns:
                X_cols.append(f"{col}_lag{lag}")
        
        # Add exogenous variables column names if present
        if self.exog is not None and self.nvar_ex > 0:
            for lag in range(self.nlag_ex):
                for col in self.exog.columns:
                    X_cols.append(f"{col}_lag{lag+1}")
        
        # Convert X to DataFrame
        X = pd.DataFrame(X, index=Y.index, columns=X_cols)
        
        # Store data matrices
        self.results['Y'] = Y
        self.results['X'] = X
        
        # Estimate equation by equation
        for j in range(self.nvar):
            y_vec = Y.iloc[:, j]
            
            # OLS estimation
            beta = np.linalg.solve(X.T @ X, X.T @ y_vec)
            yhat = X @ beta
            resid = y_vec - yhat
            
            # Compute statistics
            nobs = len(y_vec)
            k = X.shape[1]
            sigma2 = (resid.T @ resid) / (nobs - k)
            var_beta = sigma2 * np.linalg.inv(X.T @ X)
            
            # t-statistics and p-values
            stderr = np.sqrt(np.diag(var_beta))
            tstat = beta / stderr
            pval = 2 * (1 - stats.t.cdf(np.abs(tstat), nobs - k))
            
            # R-squared
            r2 = 1 - (resid.T @ resid) / ((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean()))
            r2_adj = 1 - (1 - r2) * (nobs - 1) / (nobs - k)
            
            # Store results for this equation
            eq_name = f'eq{j+1}'
            self.results[eq_name] = {
                'beta': beta,
                'stderr': stderr,
                'tstat': tstat,
                'pval': pval,
                'resid': resid.values,  # Store as numpy array
                'yhat': yhat.values,    # Store as numpy array
                'y': y_vec.values,      # Store as numpy array
                'r2': r2,
                'r2_adj': r2_adj,
                'sigma2': sigma2
            }
        
        # Compute coefficient matrices
        self._compute_coefficient_matrices()
        
        # Initialize other results
        self.results.update({
            'B': None,      # structural impact matrix
            'Biv': None,    # first columns of structural impact matrix
            'PSI': None,    # Wold multipliers
            'Fp': None,     # Recursive F by lag
            'IV': None      # External instruments for identification
        })
    
    def _compute_coefficient_matrices(self):
        """Compute coefficient matrices and VCV."""
        # Get dimensions
        k = self.results['X'].shape[1]
        
        # Get X and Y matrices
        X = self.results['X'].values
        Y = self.results['Y'].values
        
        # Compute Ft directly like in MATLAB: Ft = (X'*X)\(X'*Y)
        Ft = np.linalg.solve(X.T @ X, X.T @ Y)
        
        # Store coefficient matrices
        self.results['Ft'] = Ft
        self.results['F'] = Ft.T
        
        # Compute residuals
        resid = Y - X @ Ft
        
        # Store residuals
        self.results['resid'] = resid
        
        # Compute and store VCV matrix
        sigma = (resid.T @ resid) / (self.nobse - self.ntotcoeff)
        self.results['sigma'] = sigma
        
        # Compute companion matrix
        Fcomp = np.zeros((self.nvar * self.nlag, self.nvar * self.nlag))
        Fcomp[:self.nvar, :self.nvar*self.nlag] = self.results['F'][:, self.const:self.const+self.nvar*self.nlag]
        Fcomp[self.nvar:, :-self.nvar] = np.eye(self.nvar * (self.nlag - 1))
        self.results['Fcomp'] = Fcomp
        
        # Compute maximum eigenvalue
        self.results['maxEig'] = np.max(np.abs(np.linalg.eigvals(Fcomp))) 