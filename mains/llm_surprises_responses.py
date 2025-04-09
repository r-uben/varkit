"""
Replication of Gertler and Karadi (2015, AEJ:M) with different monetary policy instruments.

This script conducts VAR analysis comparing impulse responses to monetary policy shocks
using Cholesky identification and two instruments: 'llm' and 'ff4_tc'.

Based on:
Gertler, M., & Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity.
American Economic Journal: Macroeconomics, 7(1), 44-76.

Modified to use alternative data sources:
- Macro data: data/raw/macrodata.csv
- FF4 instrument: data/raw/fomc_surprises_jk.csv (aggregated monthly)
- SS instrument: data/gk2015/ss_surprises.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from varkit.var.var_model import VARModel
from varkit.var.var_ir import var_ir
from varkit.var.var_irband import var_irband
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from varkit.auxiliary import ols
from varkit.utils import common_sample


def load_and_process_data(macro_path: Path, ff4_path: Path, ss_path: Path, 
                          var_names: List[str], instrument_names: List[str]
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, processes, and aligns data from specified CSV files."""
    print("Loading and processing data...")

    # 1. Macro DataCPILFESL
    print(f"-> Loading macro data from {macro_path}")
    macro_cols = ["FEDFUNDS",'DGS1', 'CPILFESL', 'INDPRO', 'gdp_real', 'ebp'] # Columns to load (Corrected to DGS1)
    df_macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df_macro = df_macro[macro_cols]
    df_macro.columns = [col.lower() for col in df_macro.columns] # Lowercase names
    df_macro['cpi'] = np.log(df_macro['cpilfesl']) # Log CPI
    df_macro['ip'] = np.log(df_macro['indpro'])   # Log IP
    df_macro = df_macro.drop(columns=['cpilfesl', 'indpro']) # Drop original columns
    # Reorder columns to match var_names if necessary
    df_macro = df_macro[var_names] 
    df_macro.index = df_macro.index.to_period('M').to_timestamp('M') # Ensure monthly frequency at month end

    # 2. FF4 Instrument
    print(f"-> Loading FF4 instrument data from {ff4_path}")
    df_ff4 = pd.read_csv(ff4_path, usecols=['start', 'FF4'], parse_dates=['start'])
    df_ff4 = df_ff4.dropna(subset=['FF4']) # Drop rows where FF4 is NaN before aggregation
    # Aggregate by month - sum surprises within the month
    # Convert 'start' to month period, group by month, sum FF4, convert index back to timestamp
    ff4_monthly = df_ff4.set_index('start').resample('ME')['FF4'].sum().to_frame()
    ff4_monthly.index = ff4_monthly.index.to_period('M').to_timestamp('M') # Align index to month end
    ff4_monthly = ff4_monthly.rename(columns={'FF4': instrument_names[1]}) # Rename column

    # 3. SS Instrument (formerly llm)
    print(f"-> Loading SS instrument data from {ss_path}")
    df_ss = pd.read_csv(ss_path, index_col=0, parse_dates=True)
    df_ss.index = df_ss.index.to_period('M').to_timestamp('M') # Ensure monthly frequency at month end
    df_ss = df_ss.rename(columns={'MPI': instrument_names[0]}) # Rename column

    # 4. Combine and Align Data
    print("-> Aligning data...")
    # Use outer join first to see full range, then inner join or specific slicing
    df_combined = pd.concat([df_macro, ff4_monthly, df_ss[[instrument_names[0]]]], axis=1)
    
    # Use inner join to keep only dates where all data is available
    df_aligned = df_combined.dropna() 
    
    print(f"  Aligned data range: {df_aligned.index.min()} to {df_aligned.index.max()}")
    print(f"  Number of observations after alignment: {len(df_aligned)}")

    # Separate into endogenous and instruments
    endo_df = df_aligned[var_names]
    iv_df = df_aligned[instrument_names]

    # Standardize iv_df (each column separately)
    iv_df = (iv_df - iv_df.mean()) / iv_df.std()
    
    return endo_df, iv_df


def compute_irf_with_instrument(var_model: VARModel, instrument_data: np.ndarray, 
                              var_options: Dict) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict]:
    """Compute IRFs and bands with a specific instrument.
    
    Returns:
        tuple:
        
            - Tuple containing INF, SUP, MED, BAR arrays for IRFs
            - Updated var_results dictionary from var_ir (may include F-stats)
    """
    # Create a copy of the results to avoid modifying the original model's results dict
    results_copy = var_model.results.copy()
    # Set the instrument in the copied results
    results_copy['IV'] = instrument_data
    
    # Compute IRFs and confidence bands using the copied results
    _, var_results_iv = var_ir(results_copy, var_options)
    INF, SUP, MED, BAR = var_irband(var_results_iv, var_options)
    
    return (INF, SUP, MED, BAR), var_results_iv


def main():
    """Main script comparing impulse responses with different instruments."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths for new data sources
    macro_path = Path('data/raw/macrodata.csv')
    ff4_path = Path('data/raw/fomc_surprises_jk.csv')
    ss_path = Path('data/gk2015/ss_surprises.csv') # Using existing path for ss_surprises
    output_path = Path('data/gk2015/figures') # Keep output path or change if needed
    output_path.mkdir(parents=True, exist_ok=True)
    
    # VAR specification based on user input
    var_names = ['fedfunds', 'cpi','ip', 'ebp']  # Order matters for Cholesky (Corrected to dgs1)
    instrument_names = ['ss', 'ff4']  # The two instruments to compare
    var_nlags = 12  # Monthly data, 1 year of lags
    var_const = 1   # Include constant term
    
    # Load, process, and align data
    endo_df, iv_df = load_and_process_data(
        macro_path, ff4_path, ss_path, var_names, instrument_names
    )

    #time_window = pd.date_range(start='2000-01-01', end='2009-01-01', freq='M')
    #endo_df = endo_df.loc[endo_df.index.isin(time_window)]
    #iv_df = iv_df.loc[iv_df.index.isin(time_window)]
    #breakpoint()
  
    # Get variable and instrument names (using the lists directly now)
    nvar = len(var_names)
    # No longer need long names or mapping
    
    # Estimate VAR
    print("\nEstimating VAR model...")
    var_model = VARModel(
        endo=endo_df,
        nlag=var_nlags,
        const=var_const
    )
    print(f"  VAR estimated with {var_model.nobs} effective observations.")
    
    # --- Calculate and Print First-Stage IV Diagnostics ---
    print("\n--- Calculating First-Stage IV Diagnostics ---")
    # Helper function for safe formatting
    def format_safe(value, fmt=".4f"):
        if isinstance(value, (int, float, np.number)) and not np.isnan(value):
            return f"{value:{fmt}}"
        return "N/A"
        
    resid = var_model.results['resid']
    nlag = var_model.results['nlag']
    
    for inst_name in instrument_names:
        print(f"\nInstrument: {inst_name}")
        instrument_data = iv_df[inst_name].values.reshape(-1, 1)

        up = resid[:, 0] 
        iv_postlag = instrument_data[nlag:]
        
        if len(iv_postlag.shape) == 1:
            iv_postlag_col = iv_postlag.reshape(-1, 1)
        else:
            iv_postlag_col = iv_postlag[:, 0].reshape(-1, 1)

        if up.shape[0] != iv_postlag_col.shape[0]:
             print(f"  Warning: Residuals length ({up.shape[0]}) != post-lag IV length ({iv_postlag_col.shape[0]}). Using common_sample.")
        
        z_combined = np.column_stack([up, iv_postlag_col])
        aux, fo, lo = common_sample(z_combined, dim=0) 
        
        p = aux[:, 0]  # Matched residuals for first variable (dependent variable)
        z = aux[:, 1:]  # Matched instrument data (independent variable)
        
        first_stage_rsq = np.nan
        f_stat_standard = np.nan
        f_stat_robust = np.nan
        n_obs_fs = len(p) 
        
        if n_obs_fs < 2: 
            print("  Error: Not enough overlapping observations for first-stage regression.")
            continue 

        try:
            first_stage = ols(p, z) # add_constant=True by default
        
            # --- Manual R-squared Calculation ---
            first_stage_resid_fs = first_stage.get('resid')
            if first_stage_resid_fs is not None:
                ss_res = np.sum(first_stage_resid_fs**2)
                p_mean = np.mean(p)
                ss_tot = np.sum((p - p_mean)**2)
                if ss_tot > 1e-10: # Avoid division by zero if p is constant
                     first_stage_rsq = 1 - (ss_res / ss_tot)
                else:
                     first_stage_rsq = np.nan # R-squared is undefined if variance of p is zero
            else:
                 print("  Warning: Could not retrieve residuals to calculate R-squared.")
            # ------------------------------------
            
            # Standard F-stat
            t_stats = first_stage.get('tstat') 
            if isinstance(t_stats, (list, np.ndarray)) and len(t_stats) > 1:
                t_stat_instr_raw = t_stats[1] 
                t_stat_instr_scalar = t_stat_instr_raw[0] if isinstance(t_stat_instr_raw, (list, np.ndarray)) else t_stat_instr_raw
                try:
                    f_stat_standard = float(t_stat_instr_scalar)**2
                except (TypeError, ValueError):
                    f_stat_standard = np.nan
            
            # Robust F-stat
            betas_fs = first_stage.get('beta')
            if first_stage_resid_fs is not None and betas_fs is not None: # Check if resid was retrieved
                # Construct X matrix 
                if z.ndim == 1:
                    X_fs = np.hstack([np.ones((n_obs_fs, 1)), z.reshape(-1, 1)])
                else:
                    X_fs = np.hstack([np.ones((n_obs_fs, 1)), z])
                
                if X_fs.shape[0] != n_obs_fs: 
                     raise ValueError("X_fs row mismatch in diagnostic calculation.")
                
                k_vars_fs = X_fs.shape[1]
                if n_obs_fs > k_vars_fs: 
                    try:
                        xtx_inv = np.linalg.inv(X_fs.T @ X_fs)
                        resid_1d = first_stage_resid_fs.flatten()
                        omega_hat = np.diag(resid_1d**2)
                        
                        if X_fs.T.shape[1] == omega_hat.shape[0]: 
                            vcv_robust = xtx_inv @ (X_fs.T @ omega_hat @ X_fs) @ xtx_inv
                            
                            if vcv_robust.shape[0] > 1 and vcv_robust.shape[1] > 1:
                                se_robust_beta1_val = np.sqrt(vcv_robust[1, 1])
                                beta1_scalar = np.nan
                                if isinstance(betas_fs, (list, np.ndarray)) and len(betas_fs) > 1:
                                    beta1_raw = betas_fs[1]
                                    beta1_scalar = beta1_raw[0] if isinstance(beta1_raw, (list, np.ndarray)) else beta1_raw
                                
                                if not np.isnan(beta1_scalar) and not np.isnan(se_robust_beta1_val) and se_robust_beta1_val != 0:
                                    try:
                                        t_robust = float(beta1_scalar) / float(se_robust_beta1_val)
                                        f_stat_robust = t_robust**2
                                    except (TypeError, ValueError):
                                         f_stat_robust = np.nan 
                        else:
                             print("  Warning: Dimension mismatch calculating robust VCV.")
                    except np.linalg.LinAlgError:
                         print("  Warning: Singular matrix encountered calculating robust VCV.")
                else:
                     print("  Warning: Not enough degrees of freedom for robust VCV calculation.")
            # else: # Implicitly handled by first_stage_resid_fs check earlier
            #      print("  Warning: Could not retrieve residuals or betas for robust F-stat calculation.")

        except Exception as e:
            print(f"  Error calculating diagnostics for {inst_name}: {e}")

        # Print diagnostics
        print(f"  Observations in 1st Stage: {n_obs_fs}")
        print(f"  R-squared: {format_safe(first_stage_rsq)}") # Now should show calculated value
        print(f"  F-statistic (standard): {format_safe(f_stat_standard)}")
        print(f"  F-statistic (robust): {format_safe(f_stat_robust)}")
        if isinstance(f_stat_standard, (int, float, np.number)) and not np.isnan(f_stat_standard) and f_stat_standard < 10:
            print("  Warning: Standard F-statistic is less than 10, indicating potential weak instrument.")
            
    print("----------------------------------------------")
    # --- End Diagnostics Section ---

    # CHOLESKY IDENTIFICATION
    print("\nComputing Cholesky IRFs and Error Bands...")
    var_options = {
        'ident': 'short',  # Cholesky
        'method': 'wild',  # Wild bootstrap
        'nsteps': 48,      # 48 months
        'ndraws': 200,     # 200 bootstrap replications (adjust if needed)
        'pctg': 95,        # 95% confidence bands
        'mult': 10         # Print progress every 10 draws
    }
    
    # Compute IRFs and confidence bands with Cholesky
    # The shock is implicitly the first variable due to 'short' ident
    _, var_results_chol = var_ir(var_model.results, var_options) 
    INF_chol, SUP_chol, MED_chol, BAR_chol = var_irband(var_results_chol, var_options)
    
    # IV IDENTIFICATION
    print("\nEstimating IV IRFs and error bands for both instruments...")
    var_options_iv = var_options.copy() # Use a copy for IV options, change ident
    var_options_iv['ident'] = 'iv'
    
    # Compute IRFs for first instrument (ss)
    print(f"\n -> Instrument: {instrument_names[0]}")
    (INF1, SUP1, MED1, BAR1), _ = compute_irf_with_instrument(
        var_model, 
        iv_df[instrument_names[0]].values.reshape(-1, 1), # Pass the specific instrument data
        var_options_iv
    )
    
    # Compute IRFs for second instrument (ff4)
    print(f"\n -> Instrument: {instrument_names[1]}")
    (INF2, SUP2, MED2, BAR2), _ = compute_irf_with_instrument( # Removed var_results_ff4 capture
        var_model, 
        iv_df[instrument_names[1]].values.reshape(-1, 1), # Pass the specific instrument data
        var_options_iv
    )
    
    # Create figure with Cholesky (left) and both IV methods (right)
    print("\nGenerating plots...")
    print("-> Generating comparison plot (Cholesky vs IVs)...")
    plt.figure(figsize=(12, 10)) # Adjusted figure size
    
    # Plot all three identification approaches
    for ii in range(nvar):
        # Cholesky subplot (left column) - Response to first variable shock
        plt.subplot(nvar, 2, 2*ii + 1)
        # BAR_chol shape is (nsteps, nvar) assuming response to first shock only
        plt.plot(BAR_chol[:, ii], '-r', linewidth=2, label=f'Cholesky (Shock: {var_names[0]})') 
        plt.fill_between(np.arange(var_options['nsteps']), 
                         INF_chol[:, ii], SUP_chol[:, ii], 
                         color='r', alpha=0.2)
        plt.plot(np.zeros(var_options['nsteps']), '-k', linewidth=0.5)
        plt.title(var_names[ii], fontweight='bold')
        plt.ylabel("Response")
        if ii == nvar - 1:
            plt.xlabel("Months")
        plt.axis('tight')
        plt.legend()
        
        # IV methods subplot (right column)
        # BAR1/BAR2 shape is (nsteps, nvar) as IV identifies one shock
        plt.subplot(nvar, 2, 2*ii + 2)
        
        # First instrument (ss)
        plt.plot(BAR1[:, ii], '-b', linewidth=2, label=f'IV: {instrument_names[0]}')
        plt.fill_between(np.arange(var_options_iv['nsteps']), 
                         INF1[:, ii], SUP1[:, ii], 
                         color='b', alpha=0.1)
        
        # Second instrument (ff4)
        plt.plot(BAR2[:, ii], '-g', linewidth=2, label=f'IV: {instrument_names[1]}')
        plt.fill_between(np.arange(var_options_iv['nsteps']), 
                         INF2[:, ii], SUP2[:, ii], 
                         color='g', alpha=0.1)
        
        plt.plot(np.zeros(var_options_iv['nsteps']), '-k', linewidth=0.5)
        plt.title(var_names[ii], fontweight='bold')
        # plt.ylabel("Response") # Redundant y-label
        if ii == nvar - 1:
            plt.xlabel("Months")
        plt.axis('tight')
        plt.legend()
    
    plt.tight_layout()
    output_filename_comp = output_path / 'cholesky_vs_ivs_new_data.pdf'
    plt.savefig(output_filename_comp)
    print(f"  Saved comparison plot to {output_filename_comp}")
    plt.close() # Close the figure
    
    # Create more detailed figure with separate panels for each identification
    print("-> Generating detailed plot (separate identifications)...")
    plt.figure(figsize=(18, 10)) # Adjusted figure size
    for ii in range(nvar):
        # Cholesky subplot
        plt.subplot(nvar, 3, 3*ii + 1)
        # BAR_chol shape is (nsteps, nvar) assuming response to first shock only
        plt.plot(BAR_chol[:, ii], '-r', linewidth=2, label=f'Cholesky (Shock: {var_names[0]})')
        plt.fill_between(np.arange(var_options['nsteps']), 
                         INF_chol[:, ii], SUP_chol[:, ii], 
                         color='r', alpha=0.2)
        plt.plot(np.zeros(var_options['nsteps']), '-k', linewidth=0.5)
        plt.title(f"{var_names[ii]} - Cholesky", fontweight='bold')
        plt.ylabel("Response")
        if ii == nvar - 1:
            plt.xlabel("Months")
        plt.axis('tight')
        plt.legend()
        
        # First IV subplot (ss)
        plt.subplot(nvar, 3, 3*ii + 2)
        plt.plot(BAR1[:, ii], '-b', linewidth=2, label=f'IV: {instrument_names[0]}')
        plt.fill_between(np.arange(var_options_iv['nsteps']), 
                         INF1[:, ii], SUP1[:, ii], 
                         color='b', alpha=0.2)
        plt.plot(np.zeros(var_options_iv['nsteps']), '-k', linewidth=0.5)
        plt.title(f"{var_names[ii]} - IV: {instrument_names[0]}", fontweight='bold')
        # plt.ylabel("Response")
        if ii == nvar - 1:
            plt.xlabel("Months")
        plt.axis('tight')
        plt.legend()
        
        # Second IV subplot (ff4)
        plt.subplot(nvar, 3, 3*ii + 3)
        plt.plot(BAR2[:, ii], '-g', linewidth=2, label=f'IV: {instrument_names[1]}')
        plt.fill_between(np.arange(var_options_iv['nsteps']), 
                         INF2[:, ii], SUP2[:, ii], 
                         color='g', alpha=0.2)
        plt.plot(np.zeros(var_options_iv['nsteps']), '-k', linewidth=0.5)
        plt.title(f"{var_names[ii]} - IV: {instrument_names[1]}", fontweight='bold')
        # plt.ylabel("Response")
        if ii == nvar - 1:
            plt.xlabel("Months")
        plt.axis('tight')
        plt.legend()
    
    plt.tight_layout()
    output_filename_detail = output_path / 'three_identifications_detailed_new_data.pdf'
    plt.savefig(output_filename_detail)
    print(f"  Saved detailed plot to {output_filename_detail}")
    plt.close()
    print("\nScript finished.")


if __name__ == '__main__':
    main() 