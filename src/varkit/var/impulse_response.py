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
import statsmodels.api as sm

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

        # Align IV data with endogenous variables if using IV identification
        self._align_iv_data()

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
                self.__B = VARUtils.get_cholesky_identification_long(self.results.sigma, self.results.F_comp)
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
                up = resid.loc[:, self.options['shock_var']]     # residuals to be instrumented (1st variable)
                uq = resid.loc[:, self.results.endo.columns.drop(self.options['shock_var'])]    # residuals for second stage (other variables)

                # Step 2: Prepare instrument data
                # Make sample of IV comparable with up and uq by matching post-lag samples
                # using the CommonSample function (exactly as in MATLAB)
                # First get post-lag IV data
                iv_postlag = self.results.IV.iloc[self.results.nlag:]

                # Step 3: First stage regression (Keep this part to get p_hat)
                # Regress first variable residuals on instrument
                up, iv_postlag = self._get_common_sample(up, iv_postlag)
                first_stage = sm.OLS(up, sm.add_constant(iv_postlag)).fit()  # add_constant=True by default
                up_hat = pd.DataFrame(first_stage.predict(), index=up.index, columns=[self.options['shock_var']])  # Use statsmodels to get fitted values
                self.results.first_stage = first_stage 
                
                # Step 4: Second stage regressions to get impact responses
                B = pd.DataFrame(
                    np.zeros((self.results.nvar, self.results.nvar)),
                    index=self.results.endo.columns,
                    columns=self.results.endo.columns
                )
                B.loc[self.options['shock_var'], self.options['shock_var']] = 1  # First element normalized to 1
                
                # Get common sample for all variables at once
                uq, up_hat = self._get_common_sample(uq, up_hat)
                
                # Run second stage regression for all variables at once
                second_stage = sm.OLS(uq, sm.add_constant(up_hat)).fit()
                
                # Store coefficients directly in B matrix (skipping constant)
                B.loc[uq.columns, self.options['shock_var']] = second_stage.params[1:].values.flatten()

                # Step 5: Calculate shock size scaling factor
                # Update size of the shock following function 4 of Gertler and Karadi (2015)
                # This adjusts the shock size to account for proxy variable imperfections
                
                # Get residuals from both first and second stage
                u = pd.concat([up, uq], axis=1)

                # Calculate the variance-covariance matrix of residuals using pandas
                pq_mean = np.mean(u, axis=0)
                pq_demeaned = u - np.tile(pq_mean, (len(u), 1))  # Center the data
                sigma_b = (1/(len(u)-self.results.ntotcoeff)) * (pq_demeaned.T @ pq_demeaned)
                
                # Extract components following MATLAB notation for clarity
                # s21s11 is the vector of second stage coefficients (impact responses)
                s21s11 = B.loc[uq.columns, self.options['shock_var']]  # Column vector
                S11 = sigma_b.loc[self.options['shock_var'], self.options['shock_var']]           # Variance of first variable residuals
                S21 = sigma_b.loc[uq.columns, self.options['shock_var']] # Covariance of other vars with first var
                S22 = sigma_b.loc[uq.columns, uq.columns]        # Variance-covariance of other variables

                # Compute Q matrix following the formula in the paper, exactly as in MATLAB
                # Q = s21s11*S11*s21s11'-(S21*s21s11'+s21s11*S21')+S22
                Q = (s21s11 * S11 * s21s11.T) - (S21 @ s21s11.T + s21s11 @ S21.T) + S22
                breakpoint()
                # Compute shock scaling factor following the formula
                # sp = sqrt(S11-(S21-s21s11*S11)'*(Q\(S21-s21s11*S11)));
                S21_term = S21 - s21s11 * S11
                term = S11 - S21_term.T @ np.linalg.solve(Q, S21_term)
                
                # Silently reject draws with invalid terms by returning None
                if term <= 0 or np.isnan(term):
                    return None
                    
                sp = np.sqrt(term)
        
                # Scale B matrix by the computed factor
                B = B * sp
                self.__B = B
                breakpoint()
                
                # Store IV-specific results
                self.results.sigma_b = sigma_b
                self.results.Biv = B.loc[:, self.options['shock_var']]
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
        

        # Retrieve and initialize variables
        impact = self.options.get('impact', 0)  # Default to 0 (one stdev shock)
        recurs = self.options.get('recurs', 'wold')  # Default to 'wold'
        F_comp = self.results.F_comp
        nvar = self.results.nvar
        
        IR = {}  # Initialize dictionary for IRFs
        
        # Initialize response DataFrame
        response = pd.DataFrame(np.zeros((self.options['nsteps'], self.results.nvar)), columns=self.results.var_names)
        
        # Loop through all variables
        for var in self.results.var_names:
            # For IV identification, only compute response for shock_var
            if self.options['ident'] == 'iv' and var != self.options['shock_var']:
                continue
                
            # Get impulse vector for current variable
            impulse = VARUtils.get_unitary_shock(self.B, impact, var)
            response.loc[0, :] = (self.B @ impulse).values.flatten()
            
            # Compute remaining steps based on recursion method
            if recurs == 'wold':
                # Vectorized computation using Wold representation
                for step in range(self.options['nsteps']):
                    take_step = self.PSI.step == step
                    Psi = self.PSI.loc[take_step, self.results.var_names]
                    Psi.index = self.results.var_names
                    response.loc[step, :] = (Psi @ self.B @ impulse).values.flatten()
            else:  # recurs == 'comp'
                # Vectorized computation using companion form
                for step in range(self.options['nsteps']):
                    response.loc[step, :] = (np.linalg.matrix_power(F_comp[:nvar, :nvar], step) @ self.B @ impulse).flatten()

            IR[var] = response.copy()  # Important to copy to avoid reference issues

        # If using IV identification, ensure IR only contains shock_var response
        if self.options['ident'] == 'iv':
            IR = {self.options['shock_var']: IR[self.options['shock_var']]}
        
        return IR
    
    def _get_bootstrapped_residuals(self, IV: Optional[np.ndarray] = None):

        if self.options['method'] == 'bs':
            # Standard bootstrap: randomly sample residuals with replacement
            idx = np.random.randint(0, self.results.nobs-self.results.nlag, self.results.nobs-self.results.nlag)
            u = self.results.fit.resid[idx]
            return u, None
        elif self.options['method'] == 'wild':
            # Wild bootstrap: multiply residuals by random +1/-1
            if self.options.get('ident') == 'iv' and IV is not None:
                # For IV, we need to bootstrap both residuals and instrument
                rr = 1 - 2 * (np.random.rand(self.results.IV.shape[0], self.results.IV.shape[1]) > 0.5)
                u = self.results.fit.resid * (rr[self.results.nlag:, :] @ np.ones((self.results.IV.shape[1], self.results.nvar)))
                z = pd.DataFrame(np.vstack([
                    self.results.IV[:self.results.nlag],  # Keep first nlag observations as is
                    self.results.IV[self.results.nlag:] * rr[self.results.nlag:, :]  # Bootstrap the rest of instrument data
                ]), index=self.results.IV.index)
                return u, z
            else:
                # For Cholesky or other methods, just bootstrap residuals
                rr = 1 - 2 * (np.random.rand(self.results.fit.resid.shape[0], 1) > 0.5)
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
                - BAR: Mean response (nsteps, nvar) with outlier removal
        """
        def remove_outliers(data, n_std=3):
            """Remove outliers from bootstrap draws using z-score method.
            
            Args:
                data: DataFrame of bootstrap draws (rows=steps, columns=draws)
                n_std: Number of standard deviations to use as threshold
                
            Returns:
                DataFrame with outliers masked
            """
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            z_scores = data.sub(mean, axis=0).div(std, axis=0).abs()
            mask = z_scores < n_std
            return data.where(mask)

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
            np.zeros((nobs, nvar)), 
            columns=self.results.endo.columns, 
            index=pd.date_range(
                start=self.results.endo.index[0],
                periods=nobs,
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
            for jj in range(nlag, nobs):
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
                max_eig = max(abs(np.linalg.eigvals(var_model.results.F_comp)))

                # STEP 5: Calculate impulse responses from bootstrapped VAR
                if max_eig < 0.9999:
                    impulse_response = ImpulseResponse(var_model.results, ir_options)
                    IR_draw = impulse_response.get_impulse_response()
                    
                    IR[tt] = IR_draw[shock_var]
                    
                
                tt += 1
                pbar.update(1)  # Update progress bar only on accepted draw
            except:
                # Skip silently to avoid fly_arooding console
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
       
            if self.options["ident"] == 'iv':
                all_draws = pd.DataFrame(
                    {draw: IR[draw][var] for draw in IR.keys()},
                    index=pd.RangeIndex(start=0, stop=nsteps, name='step')
                )
            else:
                all_draws = pd.DataFrame(
                    {draw: IR[draw][var] for draw in range(tt)},
                    index=pd.RangeIndex(start=0, stop=nsteps, name='step')
                )

            # Remove outliers before computing mean (BAR)
            cleaned_draws = remove_outliers(all_draws)
            
            # Compute percentiles and mean across draws (axis=1 for row-wise operations)
            INF[var] = all_draws.quantile(pctg_inf/100, axis=1)
            SUP[var] = all_draws.quantile(pctg_sup/100, axis=1)
            MED[var] = all_draws.quantile(0.5, axis=1)
            BAR[var] = cleaned_draws.mean(axis=1)  # Use cleaned data for mean

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

    def _get_common_sample(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
        """Get common sample between two DataFrames.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
        
        Returns:
            pd.Index: Common sample index
        """
        common_sample = df1.index.intersection(df2.index)
        df1 = df1.loc[common_sample]
        df2 = df2.loc[common_sample]
        return df1, df2

    def _align_iv_data(self) -> None:
        """Align instrument variable data with endogenous variables.
        
        This function ensures that the instrument variable (IV) data and endogenous variables
        have matching samples by:
        1. Finding the common sample between IV and endogenous data using index intersection
        2. Updating both IV and endogenous data to use only the common sample
        
        This is particularly important for IV identification where both datasets need to be
        perfectly aligned for proper estimation.
        """
        if self.options['ident'] == 'iv':
            # Drop NaNs from both datasets
            iv = self.results.IV.dropna()
            endo = self.results.endo.dropna()
            # Get common sample between IV and endogenous data using index intersection
            iv, endo = self._get_common_sample(iv, endo)
            # Update both datasets to use only the common sample
            self.results.IV = iv
            self.results.endo = endo

