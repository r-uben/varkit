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

from .model import Model, Output
from ..auxiliary import ols
from ..utils import common_sample
from ..utils.var import VARUtils
# Estimate VAR on bootstrapped data
from .model import Model

class ImpulseResponse:
    def __init__(self, results: Output, options: Dict[str, Union[int, str]]):
        """Initialize ImpulseResponse with VAR model and options.

        Args:
            model: An instance of the VAR model.
            options: Dictionary of options for impulse response computation.
        """
        self.results = results
        self.fit = results.fit
        self.options = options

        self.__B = None
        self.__PSI = None
        self.__Fp = None

    @property
    def PSI(self):
        if self.__PSI is None:
            self.__PSI, self.__Fp = self.get_wold_representation()
        return self.__PSI
    
    @property
    def Fp(self):
        if self.__Fp is None:
            self.__PSI, self.__Fp = self.get_wold_representation()
        return self.__Fp

    @property
    def B(self):
        if self.__B is None:
            if self.options['ident'] == 'short':
                self.__B = VARUtils.get_cholesky_identification_short(self.results.sigma)
            elif self.options['ident'] == 'long':
                # B matrix is recovered with Cholesky on cumulative IR to infinity
                self.__B = VARUtils.get_cholesky_identification_long(self.results.sigma, self.results.Fcomp)
            elif self.options['ident'] == 'sign':
                # B matrix is recovered with sign restrictions
                if self.results.B is None:
                    raise ValueError('You need to provide the B matrix with sign restrictions')
                self.__B = self.results.B
            
            elif self.options['ident'] == 'iv':
                # B matrix is recovered with external instrument IV
                # This implementation follows Gertler and Karadi (2015) methodology
                
                # Step 1: Recover residuals (first variable is the one to be instrumented - order matters!)
                # In GK (2015), the first variable is the 1-year bond rate
                resid = self.fit.resid  # Get residuals from statsmodels
                up = resid[:, 0]     # residuals to be instrumented (1st variable)
                uq = resid[:, 1:]    # residuals for second stage (other variables)
                
                # Step 2: Prepare instrument data
                # Make sample of IV comparable with up and uq by matching post-lag samples
                # using the CommonSample function (exactly as in MATLAB)
                # First get post-lag IV data
                iv_postlag = self.results.IV[self.results.nlag:]
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
                self.results.FirstStage = first_stage 
                
                # Step 4: Second stage regressions to get impact responses
                # Recover first column of B matrix with second stage regressions
                Biv = np.zeros(self.results.nvar)
                Biv[0] = 1  # Start with impact IR normalized to 1 (by assumption)
                sqsp = np.zeros(q.shape[1])
                
                for ii in range(1, self.results.nvar):
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
                sigma_b = (1/(len(pq)-self.results.ntotcoeff)) * (pq_demeaned.T @ pq_demeaned)
                
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
                
                B = np.zeros((self.results.nvar, self.results.nvar))
                B[:, 0] = Biv  # First column is the identified shock
                
                # Store IV-specific results
                self.results.sigma_b = sigma_b
                self.results.Biv = Biv
            else:
                raise ValueError(
                    'Identification incorrectly specified.\n'
                    'Choose one of the following options:\n'
                    '- short: zero contemporaneous restrictions\n'
                    '- long:  zero long-run restrictions\n'
                    '- sign:  sign restrictions\n'
                    '- iv:  external instrument'
                )
        return self.__B



    def get_wold_representation(self) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """Compute Wold representation matrices for a VAR model.
        
        Args:
            model: VARResults object with estimation results
            nsteps: Number of steps for computation
            recurs: Method for recursive computation ('wold' or other)
        
        Returns:
            tuple:
                - PSI: DataFrame with Wold multipliers
                - Fp: Dictionary of lag coefficient matrices
        """

        if self.options['recurs'] == 'wold':
            # Get coefficients from statsmodels params DataFrame
            params = self.fit.params
            
            # Get lag coefficient matrices
            Fp = VARUtils.get_lag_coefs_matrices(params)
            
            # Compute Wold representation matrices
            PSI = VARUtils.compute_wold_matrices(Fp, self.options['nsteps'])
        else:
            # If not precomputing Wold, at least initialize PSI as DataFrame
            PSI = pd.DataFrame(
                np.zeros((self.results.nvar, self.results.nvar, self.options['nsteps'])),
                index=pd.Index(self.fit.model.endog_names, name='shock_variable'),
                columns=pd.Index(self.fit.model.endog_names, name='response_variable')
            )
            PSI.iloc[:, :, 0] = np.eye(self.results.nvar)
            Fp = None  # No need to compute Fp if not using Wold
            
        return PSI, Fp

    def get_impulse_response(self) -> np.ndarray:   

        
        """Compute impulse responses (IRs) for a VAR model.
        
        Args:
            model: VARResults object with estimation results
            options: Dictionary with VAR options
        
        Returns:
            tuple:
                - IR: array of shape (horizon, n_vars, n_vars) with impulse responses
                - model: Updated VAR results with additional fields
        """
        # Check inputs
        if self.results is None:
            raise ValueError('You need to provide VAR results')
        
        iv = self.results.IV
        if self.options['ident'] == 'iv' and iv is None:
            raise ValueError('You need to provide the data for the instrument in VAR (IV)')
        

        shock_var = self.options['shock_var']

        # Retrieve and initialize variables
        nsteps = self.options['nsteps']
        impact = self.options.get('impact', 0)  # Default to 0 (one stdev shock)
        recurs = self.options.get('recurs', 'wold')  # Default to 'wold'
        Fcomp = self.results.Fcomp
        nvar = self.results.nvar
        nlag = self.results.nlag
        sigma = self.results.sigma


        
        IR = {}  # Initialize dictionary for IRFs
        
        # Get Wold representation and compute multipliers if needed
  
        
        # Compute the impulse response
        if self.options['ident'] == 'iv':
            # For IV, only compute IRF to the identified monetary policy shock (first shock)
            mm = 0  # monetary policy shock (1-year bond rate)
            response = np.zeros((nvar, nsteps))
            impulse = np.zeros(nvar)
            
            # Set the size of the shock
            if impact == 0:
                impulse[mm] = 1  # one stdev shock to 1-year bond rate
            elif impact == 1:
                impulse[mm] = 1/self.B[mm, mm]  # unitary shock to 1-year bond rate
            else:
                raise ValueError('Impact must be either 0 or 1')
            
            # First period impulse response (=impulse vector)
            response[:, 0] = self.B @ impulse  # Response of all variables to 1-year bond rate shock
            
            # Recursive computation of impulse response
            if recurs == 'wold':
                for kk in range(1, nsteps):
                    response[:, kk] = self.PSI[:, :, kk] @ self.B @ impulse  # Propagate shock through time
            elif recurs == 'comp':
                for kk in range(1, nsteps):
                    Fcomp_n = np.linalg.matrix_power(Fcomp, kk)
                    response[:, kk] = Fcomp_n[:nvar, :nvar] @ self.B @ impulse
            
            IR[:, :, mm] = response.T  # Store responses of all variables to 1-year bond rate shock
            # Set other shocks to NaN since they're not identified
            IR[:, :, 1:] = np.nan
        else:
            # For other identification methods, compute all IRFs. But we will only use the IRF to the shock_var
            for var in self.results.var_names:
                # Initialize response matrix for all steps at once
                response = pd.DataFrame(np.zeros((nsteps, nvar)), columns=self.results.var_names)
                # Get impulse vector and compute initial response
                impulse = VARUtils.get_unitary_shock(self.B, impact, var)
                response.loc[0, :] = (self.B @ impulse).values.flatten()
                
                # Compute remaining steps based on recursion method
                if recurs == 'wold':
                    # Vectorized computation using Wold representation
                    for step in range(nsteps):
                        take_step = self.PSI.step == step
                        Psi = self.PSI.loc[take_step, self.results.var_names]
                        Psi.index = self.results.var_names
                        response.loc[step, :] = (Psi @ self.B @ impulse).values.flatten()
                    
                else:  # recurs == 'comp'
                    # Vectorized computation using companion form
                    for step in range(1, nsteps):
                        response.loc[step, :] = (np.linalg.matrix_power(Fcomp[:nvar, :nvar], step) @ self.B @ impulse).flatten()

                IR[var] = response
    
        return IR
    
    def _get_bootstrapped_residuals(self, IV: Optional[np.ndarray] = None):

        if self.options['method'] == 'bs':
            # Standard bootstrap: randomly sample residuals with replacement
            idx = np.random.randint(0, self.results.nobs, self.results.nobs)
            u = self.results.fit.resid[idx]
            return u, None
        elif self.options['method'] == 'wild':
            # Wild bootstrap: multiply residuals by random +1/-1
            if self.options.get('ident') == 'iv' and IV is not None:
                # For IV, we need to bootstrap both residuals and instrument
                rr = 1 - 2 * (np.random.rand(self.results.nobs, self.results.IV.shape[1]) > 0.5)
                u = self.results.fit.resid * (rr @ np.ones((self.results.IV.shape[1], self.results.nvar)))
                z = np.vstack([
                    self.results.IV[:self.results.nlag],  # Keep first nlag observations as is
                    self.results.IV[self.results.nlag:] * rr  # Bootstrap the rest of instrument data
                ])
                return u, z
            else:
                # For Cholesky or other methods, just bootstrap residuals
                rr = 1 - 2 * (np.random.rand(self.results.nobs, 1) > 0.5)
                u = self.results.fit.resid * (rr @ np.ones((1, self.results.nvar)))
                return u, None
        else:
            raise ValueError(f'The method {self.options["method"]} is not available')

    def get_bands(self):
        """Calculate confidence intervals for impulse response functions.
        
        Returns:
            tuple:
                - INF: Lower confidence band (nsteps, nvar)
                - SUP: Upper confidence band (nsteps, nvar)
                - MED: Median response (nsteps, nvar)
                - BAR: Mean response (nsteps, nvar)
        """
        # Check if required options are present
        required_options = ['nsteps', 'ndraws', 'pctg', 'method', 'shock_var']
        for option in required_options:
            if option not in self.options:
                raise ValueError(f'Option {option} is required for computing bands')
        
        # Retrieve options
        nsteps = self.options['nsteps']
        ndraws = self.options['ndraws']
        pctg = self.options['pctg']
        method = self.options['method']
        shock_var = self.options['shock_var']
        # Get model dimensions
        nvar = self.results.nvar
        nlag = self.results.nlag
        const = self.results.const
        nobs = len(self.fit.resid)
        resid = self.fit.resid
        
        # Get endogenous variables
        IV = self.results.IV if hasattr(self.results, 'IV') else None
        
        # Get column names
        endo_cols = self.results.endo.columns
        
        # Get the frequency information once
        original_freq = self.results.endo.index.freq or self.results.endo.index.inferred_freq
        if original_freq is None:
            # Try to infer frequency from the data
            original_freq = pd.infer_freq(self.results.endo.index)

        # Create y_artificial with proper time index
        y_artificial = pd.DataFrame(
            np.zeros((nobs + nlag, nvar)), 
            columns=self.results.endo.columns, 
            index=pd.date_range(
                start=self.results.endo.index[0],
                periods=nobs + nlag,
                freq=original_freq
            )
        )

        # Create LAG with proper frequency
        LAG = pd.DataFrame(
            np.zeros((nlag, nvar)), 
            index=pd.date_range(
                start=self.results.endo.index[0],
                periods=nlag,
                freq=original_freq
            ),
            columns=self.results.endo.columns
        )

        # Initialize storage for IRs (storing draws for all variables' responses to the first shock)
        IR = {}
        
        print(f"\nStarting {method} bootstrap with {ndraws} draws for IRF bands...")
        tt = 0  # numbers of accepted draws
        from tqdm.auto import tqdm
        pbar = tqdm(total=ndraws, desc="Bootstrap Draws")
        attempts = 0
        max_attempts = ndraws * 5  # Limit attempts to prevent infinite loops

        while tt < ndraws and attempts < max_attempts:
            attempts += 1
            # Get bootstrapped residuals
            u, z = self._get_bootstrapped_residuals(IV)
            
            # Reset values while keeping the index
            y_artificial.iloc[:] = 0
            
            # STEP 2.1: Initialize first nlag observations with real data
            for jj in range(nlag):
                y_artificial.iloc[jj] = self.results.endo.iloc[jj]
            
            # Get the coefficient matrices
            F = self.fit.params
            
            # STEP 2.2: Generate artificial series by iterating VAR equations
            for jj in range(nlag, nobs + nlag):
                # Get the lag values in correct order using DataFrame operations
                lag_data = y_artificial.iloc[jj-nlag:jj]
                # Reverse the order of lags and flatten maintaining correct structure
                X = lag_data.iloc[::-1].values.flatten()  # This reverses the order to match [t-1, t-2, ...]
                
                # Construct LAGplus using the dedicated method
                X = self._construct_lag_with_const_or_trend(X, const, jj, nlag)
                
                # Generate values for time=jj for all variables
                y_artificial.iloc[jj, :] =  F.T @ X + u.iloc[jj-nlag, :]
            
            # STEP 3: Estimate VAR on artificial bootstrapped data
            try:
                var_model = Model(
                    endo=y_artificial,
                    nlag=nlag,
                    const=const
                )
                
                # For IV, use bootstrapped instrument
                if 'z' in locals() and z is not None:
                    var_model.results.IV = z
                
                # Create options for impulse response calculation
                ir_options = self.options.copy()

                # STEP 4: Check stability of the bootstrapped VAR
                maxEig = max(abs(np.linalg.eigvals(var_model.results.Fcomp)))
                
                # STEP 5: Calculate impulse responses from bootstrapped VAR
                if maxEig < 0.9999:
                    impulse_response = ImpulseResponse(var_model.results, ir_options)
                    IR_draw = impulse_response.get_impulse_response()
                    IR[tt] = IR_draw[shock_var]
                
                tt += 1
                pbar.update(1)  # Update progress bar only on accepted draw
            except:
                # Skip silently to avoid flooding console
                continue
        pbar.close()

        if tt < ndraws:
            print(f"\nWarning: Only {tt} stable draws were accepted out of {attempts} attempts.")
        else:
            print(f"\nBootstrap finished: {tt} stable draws accepted.")
        
        # Compute the error bands from bootstrap distribution
        pctg_inf = (100 - pctg) / 2      # Lower percentile (e.g., 2.5 for 95% CI)
        pctg_sup = 100 - (100 - pctg) / 2  # Upper percentile (e.g., 97.5 for 95% CI)
        
        # Initialize DataFrames for storing results with proper index and columns
        INF = pd.DataFrame(
            np.zeros((nsteps, nvar)), 
            columns=self.results.endo.columns,
            index=pd.RangeIndex(start=0, stop=nsteps, name='step')
        )
        SUP = pd.DataFrame(
            np.zeros((nsteps, nvar)), 
            columns=self.results.endo.columns,
            index=pd.RangeIndex(start=0, stop=nsteps, name='step')
        )
        MED = pd.DataFrame(
            np.zeros((nsteps, nvar)), 
            columns=self.results.endo.columns,
            index=pd.RangeIndex(start=0, stop=nsteps, name='step')
        )
        BAR = pd.DataFrame(
            np.zeros((nsteps, nvar)), 
            columns=self.results.endo.columns,
            index=pd.RangeIndex(start=0, stop=nsteps, name='step')
        )

        # For each variable (column)
        for var in self.results.endo.columns:
            # Create a DataFrame with all draws for this variable
            # Each column is a draw, rows are time steps
            all_draws = pd.DataFrame(
                {draw: IR[draw][var] for draw in range(tt)},
                index=pd.RangeIndex(start=0, stop=nsteps, name='step')
            )
            
            # Compute percentiles and mean across draws (axis=1 for row-wise operations)
            INF[var] = all_draws.quantile(pctg_inf/100, axis=1)
            SUP[var] = all_draws.quantile(pctg_sup/100, axis=1)
            MED[var] = all_draws.quantile(0.5, axis=1)
            BAR[var] = all_draws.mean(axis=1)

        return INF, SUP, MED, BAR

    def _construct_lag_with_const_or_trend(self, X: np.ndarray, const: int, jj: int, nlag: int) -> np.ndarray:
        """Construct LAGplus based on deterministic terms.

        Args:
            LAG: Array containing the lag values.
            const: Integer indicating the type of deterministic terms to include:
                  0: No constant
                  1: Constant
                  2: Constant and linear trend
                  3: Constant, linear trend and quadratic trend
            jj: Current time index
            nlag: Number of lags in the VAR model

        Returns:
            np.ndarray: LAGplus array with deterministic terms prepended to LAG values
        """
        if const == 0:
            return X.copy()
        elif const == 1:
            return np.hstack([1, X])
        elif const == 2:
            return np.hstack([1, jj-nlag+1, X])
        elif const == 3:
            return np.hstack([1, jj-nlag+1, (jj-nlag+1)**2, X])
        else:
            raise ValueError(f"Invalid const value: {const}. Must be 0, 1, 2, or 3.")
