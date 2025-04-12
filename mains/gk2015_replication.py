"""
Replication of Gertler and Karadi (2015, AEJ:M) - Figure 1, page 61.

This script replicates the VAR analysis from:
Gertler, M., & Karadi, P. (2015). Monetary policy surprises, credit costs, and economic activity.
American Economic Journal: Macroeconomics, 7(1), 44-76.

The analysis includes both Cholesky and IV identification methods.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import font_manager
from colorama import init, Fore, Style

from varkit.var.model import Model
from varkit.var.options import Options
from varkit.var.impulse_response import ImpulseResponse
from varkit.var.plotter import VARPlotter, VARConfig

# Initialize colorama for colored console output
init(autoreset=True)

# Set up Latin Modern Roman font
plt.rcParams.update({
    'font.family': 'Latin Modern Roman',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': True,  # Enable LaTeX rendering
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Latin Modern Roman',
    'mathtext.it': 'Latin Modern Roman:italic',
    'mathtext.bf': 'Latin Modern Roman:bold'
})

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
        self._validate()
    
    def _validate(self):
        """Validate the data attributes."""
        if len(self.endo) != len(self.iv):
            raise ValueError("Endogenous and instrumental variables must have same number of observations")
        if len(self.vnames_long) != len(self.vnames_short):
            raise ValueError("Long and short variable names must have same length")
        if set(self.name_mapping.keys()) != set(self.vnames_long):
            raise ValueError("Name mapping keys must match long variable names")
        if set(self.name_mapping.values()) != set(self.vnames_short):
            raise ValueError("Name mapping values must match short variable names")
    
    @classmethod
    def from_excel(cls, data_path: Path, var_names: List[str], iv_names: List[str]) -> 'Data':
        """Create Data instance from Excel file."""
        # Read Excel file
        xl_data = pd.read_excel(data_path, header=None)
        vnames_long = [str(x).strip() for x in xl_data.iloc[0, 1:].tolist()]
        vnames_short = [str(x).strip() for x in xl_data.iloc[1, 1:].tolist()]
        
        # Load and process data
        df = pd.read_excel(data_path, skiprows=2)
        dates = df.iloc[:, 0].apply(parse_date)
        df = df.iloc[:, 1:]
        df.index = pd.DatetimeIndex(dates, freq='MS')
        df.columns = vnames_short
        
        # Create mapping and DataFrames
        name_mapping = dict(zip(vnames_long, vnames_short))
        endo_df = df[var_names]
        iv_df = (df[iv_names] - df[iv_names].mean()) / df[iv_names].std()

        return cls(
            endo=endo_df,
            iv=iv_df,
            vnames_long=vnames_long,
            vnames_short=vnames_short,
            name_mapping=name_mapping
        )
    
    def summary_stats(self) -> pd.DataFrame:
        """Calculate summary statistics for all variables."""
        stats = []
        for df in [self.endo, self.iv]:
            stats.append(df.describe())
            stats[-1]['AutoCorr'] = df.apply(lambda x: x.autocorr())
            stats[-1]['Skew'] = df.apply(lambda x: x.skew())
            stats[-1]['Kurt'] = df.apply(lambda x: x.kurtosis())
        return pd.concat(stats, axis=1)
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all variables."""
        return pd.concat([self.endo, self.iv], axis=1).corr()
    
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
        data_path = Path('data/gk2015/GK2015_Data.xlsx')
        self.data = Data.from_excel(data_path, self.config.var_names, self.config.iv_names)
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
        
        # Compute IV identification
        print(f"\n{Fore.CYAN}Estimating IV IRFs and error bands...{Style.RESET_ALL}")
        iv_results = self._compute_iv_identification(var_model)
        
        # Create plots
        print(f"\n{Fore.CYAN}Creating comparison plots...{Style.RESET_ALL}")
        output_path = Path('figures')
        self.plotter.create_comparison_plot(cholesky_results, iv_results, var_names_long, output_path)
    
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
    
    def _compute_iv_identification(self, var_model: Model) -> Dict[str, Dict]:
        """Compute IV identification for all instruments."""
        options = self._get_base_options('iv')
        results = {
            'INF': {}, 'SUP': {}, 'BAR': {}, 'MED': {}
        }
        
        for iv_var in self.config.iv_names:
            print(f"{Fore.YELLOW}Processing instrument: {iv_var}{Style.RESET_ALL}")
            
            # Get common sample
            common_sample = self.data.endo.index.intersection(
                self.data.iv.dropna().index
            )
            
            # Extend the endogenous data sample by nlags before the instrument start date
            if len(common_sample) > 0:
                start_date = common_sample[0]
                extended_start_date = start_date - pd.offsets.MonthBegin(self.config.nlags)
                extended_sample = self.data.endo.index[self.data.endo.index >= extended_start_date]
                extended_sample = extended_sample.intersection(self.data.endo.index)
            else:
                extended_sample = common_sample
                
            # Reestimate VAR with extended sample for endogenous data
            var_model_iv = Model(
                endo=self.data.endo.loc[extended_sample],
                nlag=self.config.nlags,
                const=self.config.const
            )
            var_model_iv.results.IV = self.data.iv.loc[common_sample, iv_var].to_frame()
            
            # Compute IRFs and bands
            impulse_response = ImpulseResponse(var_model_iv.results, options)
            IRiv = impulse_response.get_impulse_response()
            INF, SUP, MED, BAR = impulse_response.get_bands()
            
            # Store results
            results['INF'][iv_var] = INF
            results['SUP'][iv_var] = SUP
            results['MED'][iv_var] = MED
            results['BAR'][iv_var] = BAR
        
        return results


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
        """Create default configuration for GK2015 replication."""
        return cls(
            var_names=['gs1', 'logcpi', 'logip', 'ebp'],  # Order matters for Cholesky
            shock_var='gs1',
            iv_names=['ff4_tc'],
            nlags=12,  # Monthly data, 1 year of lags
            const=1,   # Include constant term
            nsteps=48, # 48 months horizon
            ndraws=200, # Bootstrap replications
            confidence_level=95 # Confidence bands percentage
        )


def main():
    """Main replication script."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create configuration
    config = VARConfig.default_config()
    
<<<<<<< HEAD
    # VAR specification
    var_names = ['gs1', 'logcpi', 'logip', 'ebp']  # Order matters for Cholesky
    iv_names = ['ss']
    var_nlags = 12  # Monthly data, 1 year of lags
    var_const = 1   # Include constant term
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = Data.from_excel(data_path, var_names, iv_names)
    
    # Get long names for plotting
    var_names_long = data.get_long_names(var_names)
    
    # Estimate VAR
    print("\nEstimating VAR model...")
    var_model = VARModel(
        endo=data.endo,
        nlag=var_nlags,
        const=var_const
    )
    
    # CHOLESKY IDENTIFICATION
    print("\nEstimating Cholesky IRFs and error bands...")
    var_options = {
        'ident': 'short',  # Cholesky
        'method': 'wild',  # Wild bootstrap
        'nsteps': 48,      # 48 months
        'ndraws': 200,     # 200 bootstrap replications
        'pctg': 95,        # 95% confidence bands
        'mult': 10         # Print progress every 10 draws
    }
    
    # Compute IRFs and confidence bands with Cholesky
    IR, var_results = var_ir(var_model.results, var_options)
    
    INF, SUP, MED, BAR = var_irband(var_results, var_options)

    # IV IDENTIFICATION
    print("\nEstimating IV IRFs and error bands...")
    var_options['ident'] = 'iv'
    var_options['method'] = 'wild'
    
    # Add IV data to VAR results
    var_model.results['IV'] = data.iv.values

    # Compute IRFs and confidence bands with IV identification
    IRiv, var_results_iv = var_ir(var_model.results, var_options)
    INFiv, SUPiv, MEDiv, BARiv = var_irband(var_results_iv, var_options)
    
    # Create figure with both Cholesky and IV
    plt.figure(figsize=(20, 20))
    
    # Plot both identifications
    for ii in range(data.nvar):
        # Cholesky subplot (left column)
        plt.subplot(4, 2, 2*ii + 1)
        plt.plot(BAR[:, ii], '-r', linewidth=2, label='Cholesky')
        plt.plot(INF[:, ii], '--r', linewidth=1)
        plt.plot(SUP[:, ii], '--r', linewidth=1)
        plt.plot(np.zeros(var_options['nsteps']), '-k')
        plt.title(var_names_long[ii], fontweight='bold')
        plt.axis('tight')
        plt.legend()
        
        # IV subplot (right column)
        plt.subplot(4, 2, 2*ii + 2)
        plt.plot(BARiv[:, ii], '-k', linewidth=2, label='IV')
        plt.plot(INFiv[:, ii], '--k', linewidth=1)
        plt.plot(SUPiv[:, ii], '--k', linewidth=1)
        plt.plot(np.zeros(var_options['nsteps']), '-k')
        plt.title(var_names_long[ii], fontweight='bold')
        plt.axis('tight')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / f'irf_{iv_names[0]}.pdf')
    plt.close()

=======
    # Run analysis
    analysis = VARAnalysis(config)
    analysis.run_analysis()
>>>>>>> feature/optimize-var-estimation

if __name__ == '__main__':
    main() 