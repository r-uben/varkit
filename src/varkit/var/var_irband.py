"""
Calculate confidence intervals for impulse response functions.

This module implements bootstrap-based confidence intervals for impulse responses,
supporting both standard and wild bootstrap methods.
"""

import numpy as np
from typing import Dict, Tuple, Union
import pandas as pd
from tqdm.auto import tqdm
from .var_model import VARModel, VARResults
from .var_ir import var_ir


def var_irband(var_results: VARResults, var_options: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate confidence intervals for impulse response functions.
    
    Args:
        var_results: VARResults object with estimation results
        var_options: Dictionary with VAR options
    
    Returns:
        tuple:
            - INF: Lower confidence band (nsteps, nvar, nvar)
            - SUP: Upper confidence band (nsteps, nvar, nvar)
            - MED: Median response (nsteps, nvar, nvar)
            - BAR: Mean response (nsteps, nvar, nvar)
    """
    # Check inputs
    if var_results is None:
        raise ValueError('You need to provide VAR results')
    if not var_options:
        raise ValueError('You need to provide VAR options')
    
    # Retrieve and initialize variables
    nsteps = var_options['nsteps']
    ndraws = var_options['ndraws']
    pctg = var_options['pctg']
    method = var_options['method']
    
    # Convert Ft to numpy array if it's a pandas DataFrame
    Ft = var_results.Ft.values if isinstance(var_results.Ft, pd.DataFrame) else var_results.Ft
    nvar = var_results.nvar
    nvar_ex = var_results.nvar_ex
    nlag = var_results.nlag
    nlag_ex = var_results.nlag_ex
    const = var_results.const
    nobs = var_results.nobs
    resid = var_results.var_results.resid  # Get residuals from statsmodels results
    
    # Get the data from VARResults
    ENDO = var_results.ENDO
    EXOG = var_results.EXOG
    IV = var_results.IV
    
    # Get column names from original data
    sm_var_results = var_results.var_results
    endo_cols = sm_var_results.names if hasattr(sm_var_results, 'names') else None
    exog_cols = None  # We'll handle this based on whether EXOG is a DataFrame
    
    # Create the matrices for the loop
    y_artificial = np.zeros((nobs + nlag, nvar))
    
    # Initialize storage for IRs
    IR = np.zeros((nsteps, nvar, nvar, ndraws))
    
    print(f"\nStarting {method} bootstrap with {ndraws} draws for IRF bands...")
    tt = 0  # numbers of accepted draws
    # Use tqdm for progress bar
    pbar = tqdm(total=ndraws, desc="Bootstrap Draws")
    attempts = 0
    max_attempts = ndraws * 5 # Limit attempts to prevent infinite loops

    while tt < ndraws and attempts < max_attempts:
        attempts += 1
        # STEP 1: choose the method and generate the bootstrapped residuals
        if method == 'bs':
            # Standard bootstrap: randomly sample residuals with replacement
            # This preserves contemporaneous correlation but ignores time dependence
            idx = np.random.randint(0, resid.shape[0], nobs)
            u = resid[idx]
        elif method == 'wild':
            # Wild bootstrap: multiply residuals by random +1/-1 
            # This preserves both contemporaneous and conditional heteroskedasticity
            if var_options.get('ident') == 'iv' and IV is not None:
                # For IV, we need to bootstrap both residuals and instrument
                # Use Rademacher distribution (random +1/-1) multiplier for each observation
                rr = 1 - 2 * (np.random.rand(nobs, IV.shape[1]) > 0.5)  # Create random +1/-1 matrix
                u = resid * (rr @ np.ones((IV.shape[1], nvar)))  # Apply to residuals
                z = np.vstack([
                    IV[:nlag],  # Keep first nlag observations as is
                    IV[nlag:] * rr  # Bootstrap the rest of instrument data
                ])
            else:
                # For Cholesky or other methods, just bootstrap residuals
                rr = 1 - 2 * (np.random.rand(nobs, 1) > 0.5)  # Create random +1/-1 vector
                u = resid * (rr @ np.ones((1, nvar)))  # Apply to all variables' residuals
        else:
            raise ValueError(f'The method {method} is not available')
        
        # STEP 2: generate the artificial data using bootstrapped residuals
        # STEP 2.1: initialize first nlag observations with real data
        LAG = np.array([])
        for jj in range(nlag):
            y_artificial[jj] = ENDO[jj]
            LAG = np.hstack([y_artificial[jj], LAG])
        
        # STEP 2.2: generate artificial series by iterating VAR equations
        # From observation nlag+1 to nobs, compute the artificial data using
        # estimated coefficients and bootstrapped residuals
        for jj in range(nlag, nobs + nlag):
            # Construct LAGplus vector based on deterministic terms
            if const == 0:
                LAGplus = LAG.copy()  # No constant
            elif const == 1:
                LAGplus = np.hstack([1, LAG])  # Add constant term
            elif const == 2:
                LAGplus = np.hstack([1, jj-nlag+1, LAG])  # Add constant and trend
            elif const == 3:
                LAGplus = np.hstack([1, jj-nlag+1, (jj-nlag+1)**2, LAG])  # Add constant, trend, trend²
            
            # Add exogenous variables if present
            if nvar_ex > 0 and EXOG is not None:
                if jj-nlag < len(EXOG):  # Check if within bounds
                    LAGplus = np.hstack([LAGplus, EXOG[jj-nlag]])
            
            # Generate values for time=jj for all variables using VAR equations
            breakpoint()
            for mm in range(nvar):
                y_artificial[jj, mm] = LAGplus @ Ft[:, mm] + u[jj-nlag, mm]
            
            # Update LAG matrix for next iteration
            if jj < nobs + nlag - 1:
                LAG = np.hstack([y_artificial[jj], LAG[:(nlag-1)*nvar]])
        
        # STEP 3: Estimate VAR on artificial bootstrapped data
        try:
            # Convert y_artificial to DataFrame with proper column names
            if endo_cols is not None:
                y_artificial_df = pd.DataFrame(y_artificial, columns=endo_cols)
            else:
                # Create default column names if none available
                y_artificial_df = pd.DataFrame(y_artificial, columns=[f'var{i+1}' for i in range(nvar)])
            
            # Handle exogenous variables for bootstrapped VAR
            exog_df = None
            if EXOG is not None:
                if isinstance(EXOG, pd.DataFrame):
                    # If EXOG is a DataFrame, use its column names
                    exog_df = pd.DataFrame(EXOG, columns=EXOG.columns)
                else:
                    # Create default column names if none available
                    exog_cols = [f'exog{i+1}' for i in range(EXOG.shape[1])]
                    exog_df = pd.DataFrame(EXOG, columns=exog_cols)
            
            # Estimate VAR on bootstrapped data
            var_model = VARModel(
                endo=y_artificial_df,
                nlag=nlag,
                const=const,
                exog=exog_df,
                nlag_ex=nlag_ex
            )
            
            # For IV, use bootstrapped instrument
            if 'z' in locals() and z is not None:
                var_model.results.IV = z
            
            # STEP 4: Calculate impulse responses from bootstrapped VAR
            IR_draw, var_draw = var_ir(var_model.results, var_options)
            
            # Only accept stable VARs (eigenvalues less than 1)
            if var_draw.maxEig < 0.9999:
                IR[:, :, :, tt] = IR_draw
                tt += 1
                pbar.update(1) # Update progress bar only on accepted draw
        except np.linalg.LinAlgError as e:
            # Skip silently to avoid flooding console
            continue
        except Exception as e:
            print(f"\nError during bootstrap draw {attempts}: {e}. Skipping.")
            continue

    pbar.close()

    if tt < ndraws:
        print(f"\nWarning: Only {tt} stable draws were accepted out of {attempts} attempts.")
        # Consider only accepted draws for band calculation
        IR = IR[:, :, :, :tt]
    else:
        print(f"\nBootstrap finished: {tt} stable draws accepted.")
    
    # Compute the error bands from bootstrap distribution
    pctg_inf = (100 - pctg) / 2      # Lower percentile (e.g., 2.5 for 95% CI)
    pctg_sup = 100 - (100 - pctg) / 2  # Upper percentile (e.g., 97.5 for 95% CI)
    
    # Create the full output arrays with the original 3D shape to match MATLAB
    INF_full = np.zeros((nsteps, nvar, nvar))
    SUP_full = np.zeros((nsteps, nvar, nvar))
    MED_full = np.zeros((nsteps, nvar, nvar))
    BAR_full = np.zeros((nsteps, nvar, nvar))
    
    # Focus on responses to the first shock (1-year bond rate)
    # This is the shock that is properly identified in either Cholesky or IV
    shock_idx = 0  # Index of the shock we're interested in (first variable)
    
    # For each variable, compute bands for its response to the first shock
    for ii in range(nvar):  # Loop over responding variables
        # Get all bootstrap replications of variable ii's response to first shock
        responses = IR[:, ii, shock_idx, :]  # (nsteps, ndraws)
        
        # Compute percentiles and mean across bootstrap replications
        INF_full[:, ii, shock_idx] = np.percentile(responses, pctg_inf, axis=1)
        SUP_full[:, ii, shock_idx] = np.percentile(responses, pctg_sup, axis=1)
        MED_full[:, ii, shock_idx] = np.percentile(responses, 50, axis=1)
        BAR_full[:, ii, shock_idx] = np.mean(responses, axis=1)
    
    # For backwards compatibility, also create simplified 2D arrays
    # These contain each variable's response to the first shock
    # Note: This is a key difference from MATLAB which returns full 3D arrays
    INF = INF_full[:, :, shock_idx]
    SUP = SUP_full[:, :, shock_idx]
    MED = MED_full[:, :, shock_idx]
    BAR = BAR_full[:, :, shock_idx]
    
    return INF, SUP, MED, BAR 