"""
Vector Autoregression (VAR) Model Implementation.

This module implements the main VAR model class, corresponding to the original VARmodel.m
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any


from varkit.utils.var import VARUtils

@dataclass
class Options:
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



@dataclass
class Output:
    """Class to hold VAR estimation results."""
    endo: pd.DataFrame
    exog: Optional[pd.DataFrame]
    nvar: int
    nvar_ex: int
    nlag: int
    nlag_ex: int
    const: int
    nobs: int
    ncoeff: int
    ncoeff_ex: int
    ntotcoeff: int
    Y: pd.DataFrame
    X: pd.DataFrame
    fit: Any  # statsmodels VAR results
    F: np.ndarray
    sigma: np.ndarray
    F_comp: np.ndarray
    max_eig: float
    var_names: List[str]
    B: Optional[np.ndarray] = None      # structural impact matrix
    Biv: Optional[np.ndarray] = None    # first columns of structural impact matrix
    PSI: Optional[np.ndarray] = None    # Wold multipliers
    Fp: Optional[np.ndarray] = None     # Recursive F by lag
    IV: Optional[np.ndarray] = None     # External instruments for identification
    first_stage: Optional[Any] = None   # First stage of IV identification


class Model:
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

        self._initialize_attributes(endo, nlag, const, exog, nlag_ex)
        self.options = Options()
        self.__results = None
    
    def _initialize_attributes(self, endo: pd.DataFrame, nlag: int, 
                             const: int, exog: Optional[pd.DataFrame], nlag_ex: int) -> None:
        """Initialize model attributes."""
        self.endo = endo
        self.nlag = nlag
        self.const = const
        self.exog = exog
        self.nlag_ex = nlag_ex
        
        self.nobs, self.nvar = endo.shape
        self.nvar_ex = exog.shape[1] if exog is not None else 0
        self.nobse = self.nobs - max(self.nlag, self.nlag_ex)
        
        self.ncoeff = self.nvar * self.nlag
        self.ncoeff_ex = self.nvar_ex * (self.nlag_ex + 1)
        self.ntotcoeff = self.ncoeff + self.ncoeff_ex + self.const

        self.__results = None
    
    def _validate_inputs(self) -> None:
        """Validate input data."""
        if self.exog is not None and len(self.exog) != len(self.endo):
            raise ValueError('Endogenous and exogenous variables must have same number of observations')
    
    @property
    def results(self) -> Output:
        """Get the model."""
        if self.__results is None:
            self._validate_inputs()
            self._estimate_model()
        return self.__results

    def _estimate_model(self) -> None:
        """Estimate VAR model using statsmodels."""
        # Fit VAR model
        model = VAR(self.endo)
        trend_order = VARUtils.get_trend_order(self.const)
        fit = model.fit(maxlags=self.nlag, trend=trend_order)
        
        # Prepare data matrices
        Y = pd.DataFrame(
            fit.endog[self.nlag:],
            index=self.endo.index[self.nlag:],
            columns=self.endo.columns
        )
        X = pd.DataFrame(
            fit.exog,
            index=self.endo.index[self.nlag:],
            columns=fit.exog_names
        )
        
        # Get coefficients and compute companion matrix
        var_coefs = fit.params.values
        F_comp = VARUtils.compute_companion_matrix(var_coefs, self.nvar, self.nlag)
    
        # Store results
        self.__results = Output(
            endo=self.endo,
            exog=self.exog if self.exog is not None else None,
            nvar=self.nvar,
            nvar_ex=self.nvar_ex,
            nlag=self.nlag,
            nlag_ex=self.nlag_ex,
            const=self.const,
            nobs=self.nobse,
            ncoeff=self.ncoeff,
            ncoeff_ex=self.ncoeff_ex,
            ntotcoeff=self.ntotcoeff,
            Y=Y,
            X=X,
            fit=fit,
            F=fit.params,
            sigma=fit.sigma_u,
            F_comp=F_comp,
            max_eig=np.max(np.abs(np.linalg.eigvals(F_comp))),
            var_names=fit.model.endog_names
        )
