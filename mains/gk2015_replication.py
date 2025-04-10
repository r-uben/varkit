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
from typing import Optional, Dict, List
from varkit.var.var_model import VARModel
from varkit.var.var_ir import var_ir
from varkit.var.var_irband import var_irband
import matplotlib.pyplot as plt


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
        df.index = dates
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


def main():
    """Main replication script."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Paths
    data_path = Path('data/gk2015/GK2015_Data.xlsx')
    output_path = Path('data/gk2015/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
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
    breakpoint()
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


if __name__ == '__main__':
    main() 