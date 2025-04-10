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
        """Extract lag coefficient matrices from the VAR parameter DataFrame.
        
        This function takes the estimated VAR parameters and organizes them into separate
        coefficient matrices for each lag. The resulting matrices have the following properties:
        - Each Fp[lag] is a DataFrame of shape (nvar × nvar)
        - Rows represent the response variables (the dependent variables in the VAR equations)
        - Columns represent the predictor variables (the lagged variables that appear on the right side)
        - Fp[lag][i,j] represents the effect of variable j lagged by 'lag' periods on variable i
        
        Note: The orientation of these matrices is important for correct matrix multiplication
        in the Wold representation calculation.
        
        Args:
            params: Parameter DataFrame from VAR estimation
            
        Returns:
            Dict[int, pd.DataFrame]: Dictionary mapping lag numbers to coefficient matrices
        """
        nvar = VARUtils.retrieve_nvar_from_params(params)
        nlag = VARUtils.retrieve_nlag_from_params(params)
        
        # Create coefficient matrices organized by lags as DataFrames
        Fp = {}  # Dictionary to store lag DataFrames
        for lag in range(1, nlag + 1):
            # Get coefficients for this lag across all variables
            lag_index = f'L{lag}.' # the dot is important, THINK ABOUT L11
            # Select rows where index starts with lag_index
            # Transpose to get right orientation for our matrix multiplications
            lag_coeffs = params.loc[params.index.str.startswith(lag_index)].values.T
            
            # Reshape to nvar x nvar matrix where:
            # - Rows = response variables (dependent variables in VAR equations)
            # - Columns = predictor variables (lag variables on right side of equations)
            Fp[lag] = pd.DataFrame(
                lag_coeffs.reshape(nvar, nvar),
                index=params.columns,
                columns=params.columns
            )
        return Fp

    @staticmethod
    def compute_wold_matrices(Fp: Dict[int, pd.DataFrame], nsteps: int) -> pd.DataFrame:
        """Compute Wold moving average representation matrices.
        
        The Wold representation expresses a VAR model as an infinite MA process:
        y_t = ε_t + Ψ₁ε_{t-1} + Ψ₂ε_{t-2} + ...
        
        This function computes the Ψ (PSI) matrices recursively using the formula:
        Ψ₀ = I (identity matrix)
        Ψₛ = ∑ᵢ₌₁ᵖ Ψₛ₋ᵢFᵢ for s > 0, where p is the VAR lag order
        
        Matrix orientation:
        - Each PSI[step] has rows = response variables, columns = shock variables
        - PSI[step][i,j] represents the effect of a unit shock to variable j 
          at time t on variable i at time t+step
        
        This orientation is specifically designed for computing impulse responses via:
        IR_step = PSI[step] @ B @ impulse
        
        Args:
            Fp: Dictionary of lag coefficient matrices from get_lag_coefs_matrices
            nsteps: Number of steps to compute
            
        Returns:
            pd.DataFrame: MultiIndex DataFrame with all PSI values, where:
                - 'step' index represents the time step
                - 'response_variable' index represents the variable receiving the shock effect
                - columns represent the variables from which shocks originate
        """
        nvar = Fp[1].shape[0]  # Number of variables
        var_names = Fp[1].columns  # Extract variable names from the first lag matrix
        nlag = len(Fp)  # Total number of lags present in the Fp dictionary
        
        # Initialize PSI as a list to hold DataFrames for each step
        # Each PSI[step] will have shape (nvar × nvar) with same orientation as Fp matrices
        PSI = [pd.DataFrame(np.zeros((nvar, nvar)), index=var_names, columns=var_names) for _ in range(nsteps)]
        
        # The first step is initialized as the identity matrix (Ψ₀ = I)
        PSI[0] = pd.DataFrame(np.eye(nvar), index=var_names, columns=var_names)
        
        # Compute multipliers for each step from 1 to nsteps-1 using the recursive formula
        # Ψₛ = ∑ᵢ₌₁ᵖ Ψₛ₋ᵢFᵢ for s > 0
        for step in range(1, nsteps):
            PSI[step] = sum(
                PSI[step - lag - 1] @ Fp[lag + 1]  # Matrix multiplication maintains orientation
                for lag in range(min(step, nlag))
                )
        
        # Create a MultiIndex DataFrame for easier analysis of PSI values
        # This reorganizes the PSI matrices while preserving their information
        PSI = pd.DataFrame(
            # Stack all PSI matrices into a (nsteps*nvar) × nvar array
            np.array([PSI[step].values for step in range(nsteps)]).reshape(nsteps * nvar, nvar),
            # Columns still represent shock variables (source of shocks)
            columns=pd.Index(var_names, name=''),
            # Create MultiIndex for rows with (step, response_variable)
            index=pd.MultiIndex.from_product(
                [range(nsteps), var_names],
                names=['step', 'response_variable']
            )
        )
        PSI = PSI.reset_index()
        
        # Return the MultiIndex DataFrame with reset index for better usability
        # When used for impulse responses, we will extract specific steps with:
        # PSI.loc[PSI['step'] == step, var_names]
        return PSI

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
            
            # Step 3: Create the DataFrame for B using theCholesky matrix. In this case, we have rows = response variables, columns = shock variables, so we must NOT transpose the cholesky matrix for the first variable to contemporaneously affcet all other variables.
            B = pd.DataFrame(cholesky_decomp, index=sigma.index, columns=sigma.columns)
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
        """Create an impulse vector for computing impulse responses.
        
        This function creates a vector that represents a shock to a specific variable.
        The shock size is determined by the 'impact' parameter:
        - impact=0: one standard deviation shock (uses the B matrix directly)
        - impact=1: unitary shock (scales by 1/B[shock_var,shock_var])
        
        The resulting impulse vector is used in impulse response calculations:
        IR_step = PSI[step] @ B @ impulse
        
        Args:
            B: Structural identification matrix (e.g., Cholesky decomposition of variance-covariance)
            impact: Type of shock (0=one std dev, 1=unitary)
            shock_var: Name of the variable to shock
            
        Returns:
            pd.DataFrame: Impulse vector for the specified shock
        """
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