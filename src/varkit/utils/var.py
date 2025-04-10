import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

class VARUtils:
    """Utility functions for VAR models."""

    @staticmethod
    def get_trend_order(const: int) -> str:
        """Get the trend order for the VAR model.
        
        Args:
            const: Integer indicating the type of deterministic terms
                  0: no constant
                  1: constant only
                  2: constant and trend
                  3: constant and quadratic trend
        
        Returns:
            str: Trend order specification for statsmodels
        """
        trend_orders = {
            0: -1,    # No deterministic terms
            1: 'c',   # Constant only
            2: 'ct',  # Constant and trend
            3: 'ctt'  # Constant and quadratic trend
        }
        return trend_orders[const]

    @staticmethod
    def compute_companion_matrix(coef: np.ndarray, nvar: int, nlag: int) -> np.ndarray:
        """Compute the companion matrix for the VAR model.
        
        The companion matrix transforms a VAR(p) into a VAR(1) in a higher dimension.
        For a VAR with n variables and p lags, it creates an (n*p)×(n*p) matrix:
        
        | A₁ A₂ ... Aₚ₋₁ Aₚ |
        | I  0  ... 0    0  |
        | 0  I  ... 0    0  |
        | ⋮  ⋮  ⋱  ⋮    ⋮  |
        | 0  0  ... I    0  |
        
        Args:
            coef: Coefficient matrix from statsmodels ((n*p + const) × n)
            nvar: Number of variables (n)
            nlag: Number of lags (p)
            
        Returns:
            np.ndarray: Companion matrix ((n*p) × (n*p))
        """
        n_companion = nvar * nlag
        companion = np.zeros((n_companion, n_companion))
        
        # Remove constant term if present (first row)
        var_coefs = coef[1:].T if coef.shape[0] > nvar * nlag else coef.T
            
        # Fill in the first block row with VAR coefficients
        companion[:nvar, :] = var_coefs
        
        # Fill in the identity matrices in the lower blocks
        if nlag > 1:
            idx = np.arange(nvar, n_companion)
            companion[idx[:, None], idx-nvar] = np.eye(nvar * (nlag - 1))
        
        return companion
    
    @staticmethod
    def retrieve_nlag_from_params(params):
        nlag = [x.replace('L','').split('.')[0] for x in params.index if x.startswith('L')]
        nlag = len(set(nlag))
        return nlag
    
    @staticmethod
    def retrieve_nvar_from_params(params):
        nvar = len(params.columns)
        return nvar


    @staticmethod
    def get_lag_coefs_matrices(params):
        nvar = VARUtils.retrieve_nvar_from_params(params)
        nlag = VARUtils.retrieve_nlag_from_params(params)
         # Create coefficient matrices organized by lags as DataFrames
        Fp = {}  # Dictionary to store lag DataFrames
        for lag in range(1, nlag + 1):
            # Get coefficients for this lag across all variables
            lag_index = f'L{lag}.' # the dot is important, THINK ABOUT L11
            lag_coeffs = params.loc[params.index.str.startswith(lag_index)].values  # Select rows where index starts with lag_index
    
            Fp[lag] = pd.DataFrame(
                lag_coeffs.reshape(nvar, nvar),
                index=params.columns,
                columns=params.columns
            )
        return Fp

    @staticmethod
    def compute_wold_matrices(Fp: Dict[int, pd.DataFrame], nsteps: int) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
        """Compute Wold moving average representation matrices.
        
        Args:
            Fp: Dictionary of lag coefficient matrices from get_lag_coefs_matrices
            nsteps: Number of steps to compute
            
        Returns:
            Tuple containing:
                - Dictionary of PSI matrices for each step
                - MultiIndex DataFrame with all PSI values
        """
        nvar = Fp[1].shape[0]  # Get dimensions from first lag matrix
        var_names = Fp[1].columns  # Get variable names
        nlag = len(Fp)  # Number of lags
        
        # Initialize PSI as dictionary of DataFrames
        PSI = {}
        
        # First step is identity matrix
        PSI[0] = pd.DataFrame(
            np.eye(nvar),
            index=var_names,
            columns=var_names
        )
        
        # Compute multipliers
        for step in range(1, nsteps):
            aux = pd.DataFrame(
                np.zeros((nvar, nvar)),
                index=var_names,
                columns=var_names
            )
            for lag in range(min(step, nlag)):
                aux += PSI[step-lag-1] @ Fp[lag+1]
            PSI[step] = aux
        
        # Create MultiIndex DataFrame for easier analysis
        PSI = pd.DataFrame(
            np.array([PSI[step].values for step in range(nsteps)]).reshape(nsteps * nvar, nvar),
            columns=pd.Index(var_names, name=''),
            index=pd.MultiIndex.from_product(
                [range(nsteps), var_names],
                names=['step', 'response_variable']
            )
        )
        
        return PSI.reset_index()

    @staticmethod
    def get_wold_multipliers(params):
        return 0

    @staticmethod
    def get_cholesky_identification_short(sigma):
        try:
            # Step 1: Extract the values from the sigma DataFrame
            sigma_values = sigma.values
            
            # Step 2: Compute the Cholesky decomposition
            cholesky_decomp = np.linalg.cholesky(sigma_values)
            
            # Step 3: Create the DataFrame for B using the transposed Cholesky matrix
            B = pd.DataFrame(cholesky_decomp.T, index=sigma.index, columns=sigma.columns)
        except np.linalg.LinAlgError:
            raise ValueError('VCV is not positive definite')
        return B
    
    @staticmethod
    def get_cholesky_identification_long(sigma, Fcomp):
        nvar = sigma.shape[0]
        Finf_big = np.linalg.inv(np.eye(len(Fcomp)) - Fcomp)
        Finf = Finf_big[:nvar, :nvar]
        D = np.linalg.cholesky(Finf @ sigma @ Finf.T)
        B = np.linalg.solve(Finf, D)
        return B
    
    @staticmethod
    def get_unitary_shock(B: pd.DataFrame, impact: int, shock_var: str) -> pd.DataFrame:

        nvar = B.shape[0]
        var_names = B.columns
        
        impulse = pd.DataFrame(np.zeros((nvar, 1)), index=var_names)
        impulse.index.name = 'shock'
        
        # Set the size of the shock
        if impact == 0:
            impulse.loc[shock_var, 0] = 1  # one stdev shock
        elif impact == 1:
            impulse.loc[shock_var, 0] = 1/B.loc[shock_var, shock_var]  # unitary shock
        else:
            raise ValueError('Impact must be either 0 or 1')
        return impulse