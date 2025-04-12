"""
Compute impulse responses (IRs) for a VAR model.

This module implements impulse response functions with four identification schemes:
- zero contemporaneous restrictions ('short')
- zero long-run restrictions ('long')
- sign restrictions ('sign')
- external instruments ('iv')
"""

from typing import Dict, Optional, Tuple
import numpy as np
from ..auxiliary import ols
from ..utils import common_sample


def var_ir(var_results: Dict, var_options: Dict) -> Tuple[np.ndarray, Dict]:
    """Compute impulse responses (IRs) for a VAR model.
    
    Args:
        var_results: Dictionary with VAR estimation results
        var_options: Dictionary with VAR options
    
    Returns:
        tuple:
            - IR: array of shape (horizon, n_vars, n_vars) with impulse responses
            - var_results: Updated VAR results with additional fields
    """
    # Check inputs
    if not var_results:
        raise ValueError('You need to provide VAR results')
    iv = var_results.get('IV')
    if var_options['ident'] == 'iv' and iv is None:
        raise ValueError('You need to provide the data for the instrument in VAR (IV)')
    
    # Retrieve and initialize variables
    nsteps = var_options['nsteps']
    impact = var_options.get('impact', 0)  # Default to 0 (one stdev shock)
    shut = var_options.get('shut', 0)  # Default to 0 (no shut)
    recurs = var_options.get('recurs', 'wold')  # Default to 'wold'
    Fcomp = var_results['Fcomp']
    nvar = var_results['nvar']
    nlag = var_results['nlag']
    sigma = var_results['sigma']
    IR = np.full((nsteps, nvar, nvar), np.nan)
    
    # Compute Wold representation
    PSI = np.zeros((nvar, nvar, nsteps))
    # Re-write F matrix to compute multipliers
    var_results['Fp'] = np.zeros((nvar, nvar, nlag))
    i = var_results['const']
    for ii in range(nlag):
        var_results['Fp'][:, :, ii] = var_results['F'][:, i:i+nvar]
        i += nvar
    
    # Compute multipliers
    PSI[:, :, 0] = np.eye(nvar)
    for ii in range(1, nsteps):
        aux = np.zeros((nvar, nvar))
        for jj in range(min(ii, nlag)):  # Only use up to nlag lags
            aux += PSI[:, :, ii-jj-1] @ var_results['Fp'][:, :, jj]
        PSI[:, :, ii] = aux
    
    # Update VAR with Wold multipliers
    var_results['PSI'] = PSI
    
    # Identification: Recover B matrix
    if var_options['ident'] == 'short':
        # B matrix is recovered with Cholesky decomposition
        try:
            B = np.linalg.cholesky(sigma).T
        except np.linalg.LinAlgError:
            raise ValueError('VCV is not positive definite')
    
    elif var_options['ident'] == 'long':
        # B matrix is recovered with Cholesky on cumulative IR to infinity
        Finf_big = np.linalg.inv(np.eye(len(Fcomp)) - Fcomp)
        Finf = Finf_big[:nvar, :nvar]
        D = np.linalg.cholesky(Finf @ sigma @ Finf.T).T
        B = np.linalg.solve(Finf, D)
    
    elif var_options['ident'] == 'sign':
        # B matrix is recovered with sign restrictions
        if 'B' not in var_results or var_results['B'] is None:
            raise ValueError('You need to provide the B matrix with sign restrictions')
        B = var_results['B']
    
    elif var_options['ident'] == 'iv':
        # B matrix is recovered with external instrument IV
        # This implementation follows Gertler and Karadi (2015) methodology
        
        # Step 1: Recover residuals (first variable is the one to be instrumented - order matters!)
        # In GK (2015), the first variable is the 1-year bond rate
        up = var_results['resid'][:, 0]     # residuals to be instrumented (1st variable)
        uq = var_results['resid'][:, 1:]    # residuals for second stage (other variables)
        
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

        # --- Diagnostics block removed ---
        
        # Store the first_stage results (e.g., coefficients, residuals) as it might be useful
        # Note: This dictionary will NOT contain F-stats or R-squared anymore.
        var_results['FirstStage'] = first_stage 
        # ---------------------------------------------

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
                      print(f"Warning: Could not convert second stage beta {beta1_ss_scalar} to float for var {ii}.")
                      # Decide how to handle this, e.g., set to NaN or raise error
                      # Biv[ii] = np.nan 
                      # sqsp[ii-1] = np.nan 
                      raise ValueError("Second stage coefficient issue.") # More strict
            else:
                 # Handle case where beta is not as expected
                 print(f"Warning: Could not extract second stage beta for var {ii}.")
                 raise ValueError("Second stage beta extraction issue.") # More strict

        # Step 5: Calculate shock size scaling factor
        # Update size of the shock following function 4 of Gertler and Karadi (2015)
        # This adjusts the shock size to account for proxy variable imperfections
        
        # Calculate the variance-covariance matrix of residuals
        pq_mean = np.mean(pq, axis=0)
        pq_demeaned = pq - np.tile(pq_mean, (len(pq), 1))  # Center the data
        sigma_b = (1/(len(pq)-var_results['ntotcoeff'])) * (pq_demeaned.T @ pq_demeaned)
        
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
        breakpoint()
        # Step 6: Rescale impact responses by shock size factor
        Biv = Biv * sp
        
        B = np.zeros((nvar, nvar))
        B[:, 0] = Biv  # First column is the identified shock
        breakpoint()
        
        # Update VAR with IV results for potential further analysis
        var_results['FirstStage'] = first_stage
        var_results['sigma_b'] = sigma_b
        var_results['Biv'] = Biv
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
        # For Cholesky, compute all IRFs but we'll only use responses to first shock
        for mm in range(nvar):
            # Set to zero a row of the companion matrix if "shut" is selected
            if shut != 0:
                Fcomp[shut-1, :] = 0
            
            # Initialize the impulse response vector
            response = np.zeros((nvar, nsteps))
            
            # Create the impulse vector
            impulse = np.zeros(nvar)
            
            # Set the size of the shock
            if impact == 0:
                impulse[mm] = 1  # one stdev shock
            elif impact == 1:
                impulse[mm] = 1/B[mm, mm]  # unitary shock
            else:
                raise ValueError('Impact must be either 0 or 1')
            
            # First period impulse response (=impulse vector)
            response[:, 0] = B @ impulse  # Response of all variables to shock mm
            
            # Shut down the response if "shut" is selected
            if shut != 0:
                response[shut-1, 0] = 0
            
            # Recursive computation of impulse response
            if recurs == 'wold':
                for kk in range(1, nsteps):
                    response[:, kk] = PSI[:, :, kk] @ B @ impulse
            elif recurs == 'comp':
                for kk in range(1, nsteps):
                    Fcomp_n = np.linalg.matrix_power(Fcomp, kk)
                    response[:, kk] = Fcomp_n[:nvar, :nvar] @ B @ impulse
            
            IR[:, :, mm] = response.T  # Store responses of all variables to shock mm
    
    # Update VAR with structural impact matrix
    var_results['B'] = B
    
    return IR, var_results 