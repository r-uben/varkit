"""
Replication of Gertler and Karadi (2015, AEJ:M) with different monetary policy instruments.

This script conducts VAR analysis comparing impulse responses to monetary policy shocks
using Cholesky identification and two instruments: 'ss' and 'ff4_tc'.

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
import matplotlib.pyplot as plt
from colorama import init, Fore, Style

from varkit.var.model import Model
from varkit.var.options import Options
from varkit.var.impulse_response import ImpulseResponse
from varkit.var.plotter import VARPlotter, VARConfig

# Initialize colorama for colored console output
init(autoreset=True)

def parse_date(date_str: str) -> pd.Timestamp:
    """Parse date string in YYYYmM format."""
    year = int(date_str[:4])
    month = int(date_str[5:])  # Skip the 'm'
    return pd.Timestamp(year=year, month=month, day=1)

@dataclass
class Data:
    """Class to hold and manage VAR data."""
    endo: pd.DataFrame
    iv: pd.DataFrame
    vnames_long: List[str]
    vnames_short: List[str]
    name_mapping: Dict[str, str]
    
    def __post_init__(self):
        """Initialize derived attributes after main attributes are set."""
        self.nobs = len(self.endo)
        self.nvar = len(self.endo.columns)
        self.nvar_ex = len(self.iv.columns)
        self.dates = self.endo.index

    @classmethod
    def from_csv(cls, macro_path: Path, ff4_path: Path, ss_path: Path, 
                var_names: List[str], instrument_names: List[str], nlag: int) -> 'Data':
        """Create Data instance from CSV files."""
        print(f"{Fore.CYAN}Loading and processing data...{Style.RESET_ALL}")
        
        # 1. Macro Data
        print(f"{Fore.YELLOW}-> Loading macro data from {macro_path}{Style.RESET_ALL}")
        macro_cols = ["FEDFUNDS", 'DGS1', 'CPILFESL', 'INDPRO', 'gdp_real', 'ebp']
        df_macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        df_macro = df_macro[macro_cols]
        df_macro.columns = [col.lower() for col in df_macro.columns]
        df_macro['cpi'] = np.log(df_macro['cpilfesl'])
        df_macro['ip'] = np.log(df_macro['indpro'])
        df_macro['logcpi'] = np.log(df_macro['cpilfesl'])
        df_macro['logip'] = np.log(df_macro['indpro'])
        df_macro['gs1'] = df_macro['dgs1']
        df_macro = df_macro.drop(columns=['cpilfesl', 'indpro'])
        df_macro = df_macro[var_names]
        df_macro.index = df_macro.index.to_period('M').to_timestamp('M')
        df_macro = df_macro.dropna()  # Drop any rows with NaN in macro data
        
        # 2. FF4 Instrument
        print(f"{Fore.YELLOW}-> Loading FF4 instrument data from {ff4_path}{Style.RESET_ALL}")
        df_ff4 = pd.read_csv(ff4_path, usecols=['start', 'FF4'], parse_dates=['start'])
        df_ff4 = df_ff4.dropna(subset=['FF4'])
        ff4_monthly = df_ff4.set_index('start').resample('ME')['FF4'].sum().to_frame()
        ff4_monthly.index = ff4_monthly.index.to_period('M').to_timestamp('M')
        ff4_monthly = ff4_monthly.rename(columns={'FF4': instrument_names[1]})
        
        # 3. SS Instrument
        print(f"{Fore.YELLOW}-> Loading SS instrument data from {ss_path}{Style.RESET_ALL}")
        df_ss = pd.read_csv(ss_path, index_col=0, parse_dates=True)
        df_ss.index = df_ss.index.to_period('M').to_timestamp('M')
        df_ss = df_ss.rename(columns={'MPI': instrument_names[0]})
        
        # 4. Combine instruments data and handle alignment
        print(f"{Fore.YELLOW}-> Aligning data...{Style.RESET_ALL}")
        
        # Combine both instruments datasets
        iv_combined = pd.concat([ff4_monthly, df_ss[[instrument_names[0]]]], axis=1)
        iv_combined = iv_combined.dropna()  # Drop rows with NaN in instruments
        
        # Get earliest date for instruments and macro data
        iv_start_date = iv_combined.index.min()
        macro_start_date = df_macro.index.min()
        iv_end_date = iv_combined.index.max()
        macro_end_date = df_macro.index.max()
        end_date = min(iv_end_date, macro_end_date)
        
        # Create required window for aligned data
        required_start_date = iv_start_date - pd.DateOffset(months=nlag)
        
        # CASE 1: Macro data starts early enough for pre-sample observations
        
        if macro_start_date <= required_start_date:
            print(f"{Fore.GREEN}Macro data has enough pre-sample observations (starting from {macro_start_date}){Style.RESET_ALL}")
            
            macro_final = df_macro.loc[required_start_date:end_date]
            iv_final = iv_combined.loc[required_start_date:end_date]
            
        # CASE 2: Macro data doesn't start early enough
        else:
            print(f"{Fore.YELLOW}Macro data doesn't have enough pre-sample observations. Using available data and adjusting instrument sample.{Style.RESET_ALL}")
            
            # Get the common sample between macro and instruments
            common_dates = df_macro.index.intersection(iv_combined.index)
            if len(common_dates) <= nlag:
                raise ValueError(f"Not enough common observations. Need at least {nlag+1} observations, but only found {len(common_dates)}")
            
            # Create aligned datasets
            macro_final = df_macro.loc[common_dates]
            iv_temp = iv_combined.loc[common_dates]
            
            # Create IV DataFrame with NaN for first nlag observations
            iv_final = iv_temp.copy()
            iv_final.iloc[:nlag, :] = np.nan
        
        # Standardize instruments (excluding NaN)
        # First compute mean and std for non-NaN values
        iv_means = iv_final.mean()
        iv_stds = iv_final.std()
        
        # Create standardized version with NaN preserved
        iv_standardized = iv_final.copy()
        for col in iv_standardized.columns:
            # Only standardize non-NaN values
            mask = ~iv_standardized[col].isna()
            if mask.any():  # Check if there are any non-NaN values
                iv_standardized.loc[mask, col] = (iv_standardized.loc[mask, col] - iv_means[col]) / iv_stds[col]
        
        # Verify alignment
        print(f"{Fore.GREEN}Data aligned successfully:{Style.RESET_ALL}")
        print(f"Macro data range: {macro_final.index.min()} to {macro_final.index.max()}, {len(macro_final)} observations")
        print(f"Instrument data range: {iv_final.index.min()} to {iv_final.index.max()}, {len(iv_final)} observations")
        print(f"First {nlag} IV observations are NaN for pre-sample")
        non_nan_iv = iv_standardized.dropna()
        print(f"Non-NaN instruments start at: {non_nan_iv.index.min()} ({len(non_nan_iv)} observations)")
        
        # Get variable names
        vnames_long = [
            "Federal Funds Rate",
            "Consumer Price Index (log)",
            "Industrial Production (log)",
            "Excess Bond Premium"
        ]
        vnames_short = var_names
        name_mapping = dict(zip(vnames_long, vnames_short))
        
        return cls(
            endo=macro_final,
            iv=iv_standardized,
            vnames_long=vnames_long,
            vnames_short=vnames_short,
            name_mapping=name_mapping
        )
    
    def get_long_names(self, short_names: List[str]) -> List[str]:
        """Get long names corresponding to given short names."""
        long_names = []
        rev_mapping = {v: k for k, v in self.name_mapping.items()}
        for name in short_names:
            long_names.append(rev_mapping[name])
        return long_names


class VARAnalysis:
    """Class for conducting VAR analysis."""
    
    def __init__(self, config: VARConfig):
        self.config = config
        self.data = None
        self.plotter = VARPlotter(config)
    
    def load_data(self) -> None:
        """Load and prepare data for analysis."""
        print(f"{Fore.CYAN}Loading and preparing data...{Style.RESET_ALL}")
        
        # Paths for data sources
        macro_path = Path('data/raw/macrodata.csv')
        ff4_path = Path('data/raw/fomc_surprises_jk.csv')
        ss_path = Path('data/gk2015/ss_surprises.csv')
        
        self.data = Data.from_csv(
            macro_path=macro_path,
            ff4_path=ff4_path,
            ss_path=ss_path,
            var_names=self.config.var_names,
            instrument_names=self.config.iv_names,
            nlag=self.config.nlags
        )
        
        
        print(f"{Fore.GREEN}Data loaded successfully{Style.RESET_ALL}")
    
    def run_analysis(self) -> None:
        """Run the complete VAR analysis."""
        if self.data is None:
            self.load_data()
        
        # Get long names for plotting
        var_names_long = self.data.get_long_names(self.config.var_names)
        
        # Estimate VAR
        print(f"\n{Fore.CYAN}Estimating VAR model...{Style.RESET_ALL}")
        var_model = self._estimate_var()
        # Compute Cholesky identification
        print(f"\n{Fore.CYAN}Estimating Cholesky IRFs and error bands...{Style.RESET_ALL}")
        cholesky_results = self._compute_cholesky_identification(var_model)
        
        # Compute IV identification for first instrument (SS)
        print(f"\n{Fore.CYAN}Estimating IV IRFs for {self.config.iv_names[0]}...{Style.RESET_ALL}")
        ss_results = self._compute_iv_identification(var_model, self.config.iv_names[0])
        
        # Compute IV identification for second instrument (FF4)
        print(f"\n{Fore.CYAN}Estimating IV IRFs for {self.config.iv_names[1]}...{Style.RESET_ALL}")
        ff4_results = self._compute_iv_identification(var_model, self.config.iv_names[1])
        
        # Create comparison plots
        print(f"\n{Fore.CYAN}Creating comparison plots...{Style.RESET_ALL}")
        output_path = Path('figures')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create detailed figure with separate panels for each identification
        self._create_detailed_plot(cholesky_results, ss_results, ff4_results, output_path)
        
        # Create figure with Cholesky vs both IV methods
        self._create_comparison_plot(cholesky_results, ss_results, ff4_results, output_path)
    
    def _estimate_var(self) -> Model:
        """Estimate VAR model."""
        return Model(
            endo=self.data.endo,
            nlag=self.config.nlags,
            const=self.config.const
        )
    
    def _get_base_options(self, ident: str) -> Dict:
        """Get base options for impulse response calculation."""
        return {
            'ident': ident,
            'method': 'wild',
            'nsteps': self.config.nsteps,
            'ndraws': self.config.ndraws,
            'pctg': self.config.confidence_level,
            'mult': 10,
            'recurs': 'wold',
            'shock_var': self.config.shock_var
        }
    
    def _compute_cholesky_identification(self, var_model: Model) -> Tuple:
        """Compute Cholesky identification."""
        options = self._get_base_options('short')
        impulse_response = ImpulseResponse(var_model.results, options)
        IR = impulse_response.get_impulse_response()
        return impulse_response.get_bands()
    
    def _compute_iv_identification(self, var_model: Model, iv_name: str) -> Tuple:
        """Compute IV identification for a specific instrument."""
        options = self._get_base_options('iv')
        
        # Get the instrument data
        iv_data = self.data.iv[[iv_name]].dropna()
        
        # Create a copy of the model results and add the specific instrument
        results_copy = var_model.results
        results_copy.IV = iv_data
        
        # Compute IRFs with the instrument
        impulse_response = ImpulseResponse(results_copy, options)
        IR = impulse_response.get_impulse_response()
        
        return impulse_response.get_bands()
    
    def _create_comparison_plot(self, cholesky_results, ss_results, ff4_results, output_path):
        """Create comparison plot with Cholesky vs both IV methods."""
        INF_chol, SUP_chol, MED_chol, BAR_chol = cholesky_results
        INF_ss, SUP_ss, MED_ss, BAR_ss = ss_results
        INF_ff4, SUP_ff4, MED_ff4, BAR_ff4 = ff4_results
        
        nvar = len(self.config.var_names)
        
        plt.figure(figsize=(12, 10))
        
        for ii in range(nvar):
            # Cholesky subplot (left column)
            plt.subplot(nvar, 2, 2*ii + 1)
            plt.plot(BAR_chol[self.config.var_names[ii]], '-r', linewidth=2, 
                     label=f'Cholesky (Shock: {self.config.shock_var})')
            plt.fill_between(range(self.config.nsteps), 
                             INF_chol[self.config.var_names[ii]], 
                             SUP_chol[self.config.var_names[ii]], 
                             color='r', alpha=0.2)
            plt.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5)
            plt.title(self.config.var_names[ii], fontweight='bold')
            plt.ylabel("Response")
            if ii == nvar - 1:
                plt.xlabel("Months")
            plt.axis('tight')
            plt.legend()
            
            # IV methods subplot (right column)
            plt.subplot(nvar, 2, 2*ii + 2)
            
            # SS instrument
            plt.plot(BAR_ss[self.config.var_names[ii]], '-b', linewidth=2, 
                     label=f'IV: {self.config.iv_names[0]}')
            plt.fill_between(range(self.config.nsteps), 
                             INF_ss[self.config.var_names[ii]], 
                             SUP_ss[self.config.var_names[ii]], 
                             color='b', alpha=0.1)
            
            # FF4 instrument
            plt.plot(BAR_ff4[self.config.var_names[ii]], '-g', linewidth=2, 
                     label=f'IV: {self.config.iv_names[1]}')
            plt.fill_between(range(self.config.nsteps), 
                             INF_ff4[self.config.var_names[ii]], 
                             SUP_ff4[self.config.var_names[ii]], 
                             color='g', alpha=0.1)
            
            plt.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5)
            plt.title(self.config.var_names[ii], fontweight='bold')
            if ii == nvar - 1:
                plt.xlabel("Months")
            plt.axis('tight')
            plt.legend()
        
        plt.tight_layout()
        output_filename = output_path / 'cholesky_vs_ivs_comparison.pdf'
        plt.savefig(output_filename)
        print(f"{Fore.GREEN}  Saved comparison plot to {output_filename}{Style.RESET_ALL}")
        plt.close()
    
    def _create_detailed_plot(self, cholesky_results, ss_results, ff4_results, output_path):
        """Create detailed plot with separate panels for each identification."""
        INF_chol, SUP_chol, MED_chol, BAR_chol = cholesky_results
        INF_ss, SUP_ss, MED_ss, BAR_ss = ss_results
        INF_ff4, SUP_ff4, MED_ff4, BAR_ff4 = ff4_results
        
        nvar = len(self.config.var_names)
        
        plt.figure(figsize=(18, 10))
        
        for ii in range(nvar):
            var_name = self.config.var_names[ii]
            
            # Cholesky subplot
            plt.subplot(nvar, 3, 3*ii + 1)
            plt.plot(BAR_chol[var_name], '-r', linewidth=2, 
                     label=f'Cholesky (Shock: {self.config.shock_var})')
            plt.fill_between(range(self.config.nsteps), 
                             INF_chol[var_name], SUP_chol[var_name], 
                             color='r', alpha=0.2)
            plt.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5)
            plt.title(f"{var_name} - Cholesky", fontweight='bold')
            plt.ylabel("Response")
            if ii == nvar - 1:
                plt.xlabel("Months")
            plt.axis('tight')
            plt.legend()
            
            # SS instrument subplot
            plt.subplot(nvar, 3, 3*ii + 2)
            plt.plot(BAR_ss[var_name], '-b', linewidth=2, 
                     label=f'IV: {self.config.iv_names[0]}')
            plt.fill_between(range(self.config.nsteps), 
                             INF_ss[var_name], SUP_ss[var_name], 
                             color='b', alpha=0.2)
            plt.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5)
            plt.title(f"{var_name} - IV: {self.config.iv_names[0]}", fontweight='bold')
            if ii == nvar - 1:
                plt.xlabel("Months")
            plt.axis('tight')
            plt.legend()
            
            # FF4 instrument subplot
            plt.subplot(nvar, 3, 3*ii + 3)
            plt.plot(BAR_ff4[var_name], '-g', linewidth=2, 
                     label=f'IV: {self.config.iv_names[1]}')
            plt.fill_between(range(self.config.nsteps), 
                             INF_ff4[var_name], SUP_ff4[var_name], 
                             color='g', alpha=0.2)
            plt.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5)
            plt.title(f"{var_name} - IV: {self.config.iv_names[1]}", fontweight='bold')
            if ii == nvar - 1:
                plt.xlabel("Months")
            plt.axis('tight')
            plt.legend()
        
        plt.tight_layout()
        output_filename = output_path / 'three_identifications_detailed.pdf'
        plt.savefig(output_filename)
        print(f"{Fore.GREEN}  Saved detailed plot to {output_filename}{Style.RESET_ALL}")
        plt.close()


@dataclass
class VARConfig:
    """Configuration for VAR analysis."""
    var_names: List[str]
    shock_var: str
    iv_names: List[str]
    nlags: int
    const: int
    nsteps: int
    ndraws: int
    confidence_level: float
    
    @classmethod
    def default_config(cls) -> 'VARConfig':
        """Create default configuration for LLM Surprises analysis."""
        return cls(
            var_names=['gs1', 'logcpi', 'logip', 'ebp'],  # Order matters for Cholesky
            shock_var='gs1',
            iv_names=['ss', 'ff4'],  # Two instruments to compare
            nlags=12,  # Monthly data, 1 year of lags
            const=1,   # Include constant term
            nsteps=48, # 48 months horizon
            ndraws=200, # Bootstrap replications
            confidence_level=95 # Confidence bands percentage
        )


def main():
    """Main script comparing impulse responses with different instruments."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create configuration
    config = VARConfig.default_config()
    
    # Run analysis
    analysis = VARAnalysis(config)
    analysis.run_analysis()


if __name__ == '__main__':
    main() 