"""
Compute impulse responses (IRs) for a VAR model.

This module implements impulse response functions with four identification schemes:
- zero contemporaneous restrictions ('short')
- zero long-run restrictions ('long')
- sign restrictions ('sign')
- external instruments ('iv')
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VARResults
from .var_model import VARResults
from ..auxiliary import ols
from ..utils import common_sample
from ..utils.var import VARUtils


def get_wold_representation(var_results: VARResults, nsteps: int, recurs: str = 'wold') -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Compute Wold representation matrices for a VAR model.
    
    Args:
        var_results: VARResults object with estimation results
        nsteps: Number of steps for computation
        recurs: Method for recursive computation ('wold' or other)
    
    Returns:
        tuple:
            - PSI: DataFrame with Wold multipliers
            - Fp: Dictionary of lag coefficient matrices
    """
    # Get statsmodels VARResults object and variable names
    sm_var_results = var_results.varfit
    nvar = var_results.nvar
    
    if recurs == 'wold':
        # Get coefficients from statsmodels params DataFrame
        params = sm_var_results.params
        
        # Get lag coefficient matrices
        Fp = VARUtils.get_lag_coefs_matrices(params)
          
        # Compute Wold representation matrices
        PSI = VARUtils.compute_wold_matrices(Fp, nsteps)
    else:
        # If not precomputing Wold, at least initialize PSI as DataFrame
        PSI = pd.DataFrame(
            np.zeros((nvar, nvar, nsteps)),
            index=pd.Index(sm_var_results.model.endog_names, name='shock_variable'),
            columns=pd.Index(sm_var_results.model.endog_names, name='response_variable')
        )
        PSI.iloc[:, :, 0] = np.eye(nvar)
        Fp = None  # No need to compute Fp if not using Wold
        
    return PSI, Fp

def var_ir(var_results: VARResults, var_options: Dict) -> Tuple[np.ndarray, VARResults]:
    """Compute impulse responses (IRs) for a VAR model.
    
    Args:
        var_results: VARResults object with estimation results
        var_options: Dictionary with VAR options
    
    Returns:
        tuple:
            - IR: array of shape (horizon, n_vars, n_vars) with impulse responses
            - var_results: Updated VAR results with additional fields
    """
    # Check inputs
    if var_results is None:
        raise ValueError('You need to provide VAR results')
    
    iv = var_results.IV
    if var_options['ident'] == 'iv' and iv is None:
        raise ValueError('You need to provide the data for the instrument in VAR (IV)')
    

    shock_var = var_options['shock_var']

    # Retrieve and initialize variables
    nsteps = var_options['nsteps']
    impact = var_options.get('impact', 0)  # Default to 0 (one stdev shock)
    recurs = var_options.get('recurs', 'wold')  # Default to 'wold'
    Fcomp = var_results.Fcomp
    nvar = var_results.nvar
    nlag = var_results.nlag
    sigma = var_results.sigma

    # Get statsmodels VARResults object
    sm_var_results: VARResults = var_results.varfit
    
    # Get variable names from statsmodels results
    var_names = sm_var_results.model.endog_names
    IR = pd.DataFrame(index=range(nsteps), columns=var_names)  # Initialize DataFrame for IRFs
    
    # Get Wold representation and compute multipliers if needed
    if var_results.PSI is None:
        PSI, Fp = get_wold_representation(var_results, nsteps, recurs)
        var_results.PSI = PSI
        var_results.Fp = Fp
    else:
        # Use precomputed PSI if available
        PSI = var_results.PSI
        Fp = var_results.Fp
    
    # Identification: Recover B matrix
    if var_options['ident'] == 'short':
        B = VARUtils.get_cholesky_identification_short(sigma)
    elif var_options['ident'] == 'long':
        # B matrix is recovered with Cholesky on cumulative IR to infinity
        B = VARUtils.get_cholesky_identification_long(sigma, Fcomp)
    elif var_options['ident'] == 'sign':
        # B matrix is recovered with sign restrictions
        if var_results.B is None:
            raise ValueError('You need to provide the B matrix with sign restrictions')
        B = var_results.B
    
    elif var_options['ident'] == 'iv':
        # B matrix is recovered with external instrument IV
        # This implementation follows Gertler and Karadi (2015) methodology
        
        # Step 1: Recover residuals (first variable is the one to be instrumented - order matters!)
        # In GK (2015), the first variable is the 1-year bond rate
        resid = sm_var_results.resid  # Get residuals from statsmodels
        breakpoint()
        up = resid[:, 0]     # residuals to be instrumented (1st variable)
        uq = resid[:, 1:]    # residuals for second stage (other variables)
        
        # Step 2: Prepare instrument data
        # Make sample of IV comparable with up and uq by matching post-lag samples
        # using the CommonSample function (exactly as in MATLAB)
        # First get post-lag IV data
        iv_postlag = iv[nlag:]
        # Combine residuals and instrument for common sample matching
        # Note: Need to handle both 1D and 2D instrument cases
        if len(iv_postlag.shape) == 1:
            z_combined = np.column_stack([up, iv_postlag])
        else:
            # Use first column if multiple instruments provided in IV array for this step
            z_combined = np.column_stack([up, iv_postlag[:, 0]]) 
            
        # Apply common_sample exactly as in MATLAB's CommonSample
        aux, fo, lo = common_sample(z_combined, dim=0)
        
        # Extract matched residuals and instrument data for first stage
        p = aux[:, 0]  # First variable residuals with matching instrument
        z = aux[:, 1:]  # The instrument data aligned with p
        
        # For other variables, trim to match p length (exactly as in MATLAB)
        q = uq[-len(p):, :]
        pq = np.column_stack([p, q]) # Combined matched residuals
        
        # Step 3: First stage regression (Keep this part to get p_hat)
        # Regress first variable residuals on instrument
        first_stage = ols(p, z)  # add_constant=True by default
        # Get fitted values from first stage (predicted p using instrument)
        p_hat = first_stage['yhat']

        # Store the first_stage results as it might be useful
        var_results.FirstStage = first_stage 
        
        # Step 4: Second stage regressions to get impact responses
        # Recover first column of B matrix with second stage regressions
        Biv = np.zeros(nvar)
        Biv[0] = 1  # Start with impact IR normalized to 1 (by assumption)
        sqsp = np.zeros(q.shape[1])
        
        for ii in range(1, nvar):
            # Regress each variable's residuals on fitted p_hat values
            second_stage = ols(q[:, ii-1], p_hat) # add_constant=True by default
            
            # Use beta[1] which is the second coefficient (after constant)
            # Handle potential array output from ols beta
            betas_ss = second_stage.get('beta')
            beta1_ss_scalar = np.nan
            if isinstance(betas_ss, (list, np.ndarray)) and len(betas_ss) > 1:
                 beta1_ss_raw = betas_ss[1]
                 beta1_ss_scalar = beta1_ss_raw[0] if isinstance(beta1_ss_raw, (list, np.ndarray)) else beta1_ss_raw
                 try:
                     Biv[ii] = float(beta1_ss_scalar)
                     sqsp[ii-1] = float(beta1_ss_scalar) # Store the coefficient
                 except (TypeError, ValueError):
                      raise ValueError(f"Could not convert second stage beta {beta1_ss_scalar} to float for var {ii}.")
            else:
                 raise ValueError(f"Could not extract second stage beta for var {ii}.")

        # Step 5: Calculate shock size scaling factor
        # Update size of the shock following function 4 of Gertler and Karadi (2015)
        # This adjusts the shock size to account for proxy variable imperfections
        
        # Calculate the variance-covariance matrix of residuals
        pq_mean = np.mean(pq, axis=0)
        pq_demeaned = pq - np.tile(pq_mean, (len(pq), 1))  # Center the data
        sigma_b = (1/(len(pq)-var_results.ntotcoeff)) * (pq_demeaned.T @ pq_demeaned)
        
        # Extract components following MATLAB notation for clarity
        # s21s11 is the vector of second stage coefficients (impact responses)
        s21s11 = sqsp.reshape(-1, 1)  # Column vector, matching MATLAB's dimensions
        S11 = sigma_b[0, 0]           # Variance of first variable residuals
        S21 = sigma_b[1:, 0].reshape(-1, 1)  # Covariance of other vars with first var
        S22 = sigma_b[1:, 1:]         # Variance-covariance of other variables
        
        # Compute Q matrix following the formula in the paper, exactly as in MATLAB
        # Q = s21s11*S11*s21s11'-(S21*s21s11'+s21s11*S21')+S22
        Q = (s21s11 * S11 * s21s11.T) - (S21 @ s21s11.T + s21s11 @ S21.T) + S22
        
        # Compute shock scaling factor following the formula
        # sp = sqrt(S11-(S21-s21s11*S11)'*(Q\(S21-s21s11*S11)));
        S21_term = S21 - s21s11 * S11
        sp = np.sqrt(S11 - S21_term.T @ np.linalg.solve(Q, S21_term))
        
        # Step 6: Rescale impact responses by shock size factor
        Biv = Biv * sp
        
        B = np.zeros((nvar, nvar))
        B[:, 0] = Biv  # First column is the identified shock
        
        # Store IV-specific results
        var_results.sigma_b = sigma_b
        var_results.Biv = Biv
    else:
        raise ValueError(
            'Identification incorrectly specified.\n'
            'Choose one of the following options:\n'
            '- short: zero contemporaneous restrictions\n'
            '- long:  zero long-run restrictions\n'
            '- sign:  sign restrictions\n'
            '- iv:  external instrument'
        )
    
    # Compute the impulse response
    if var_options['ident'] == 'iv':
        # For IV, only compute IRF to the identified monetary policy shock (first shock)
        mm = 0  # monetary policy shock (1-year bond rate)
        response = np.zeros((nvar, nsteps))
        impulse = np.zeros(nvar)
        
        # Set the size of the shock
        if impact == 0:
            impulse[mm] = 1  # one stdev shock to 1-year bond rate
        elif impact == 1:
            impulse[mm] = 1/B[mm, mm]  # unitary shock to 1-year bond rate
        else:
            raise ValueError('Impact must be either 0 or 1')
        
        # First period impulse response (=impulse vector)
        response[:, 0] = B @ impulse  # Response of all variables to 1-year bond rate shock
        
        # Recursive computation of impulse response
        if recurs == 'wold':
            for kk in range(1, nsteps):
                response[:, kk] = PSI[:, :, kk] @ B @ impulse  # Propagate shock through time
        elif recurs == 'comp':
            for kk in range(1, nsteps):
                Fcomp_n = np.linalg.matrix_power(Fcomp, kk)
                response[:, kk] = Fcomp_n[:nvar, :nvar] @ B @ impulse
        
        IR[:, :, mm] = response.T  # Store responses of all variables to 1-year bond rate shock
        # Set other shocks to NaN since they're not identified
        IR[:, :, 1:] = np.nan
    else:
        # For other identification methods, compute all IRFs
        for shock_var in var_names:
            # Initialize response matrix for all steps at once
            response = np.zeros((nsteps, nvar))
            
            # Get impulse vector and compute initial response
            impulse = VARUtils.get_unitary_shock(B, impact, shock_var)
            response[0] = (B @ impulse).values.flatten()
            
            # Compute remaining steps based on recursion method
            if recurs == 'wold':
                # Vectorized computation using Wold representation
                for step in range(1, nsteps):
                    Psi = PSI.loc[PSI['step'] == step, var_names]
                    Psi.index = var_names
                    response[step] = (Psi @ B @ impulse).values.flatten()
            else:  # recurs == 'comp'
                # Vectorized computation using companion form
                for step in range(1, nsteps):
                    response[step] = (np.linalg.matrix_power(Fcomp[:nvar, :nvar], step) @ B @ impulse).flatten()
            
            IR[shock_var] = response
 
    # Update VAR with structural impact matrix
    var_results.B = B
  
    return IR, var_results 