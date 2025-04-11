"""
Plotting utilities for VAR analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init(autoreset=True)

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
            iv_names=['ff4_tc', 'ss'],
            nlags=12,  # Monthly data, 1 year of lags
            const=1,   # Include constant term
            nsteps=48, # 48 months horizon
            ndraws=200, # Bootstrap replications
            confidence_level=95 # Confidence bands percentage
        )

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

class VARPlotter:
    """Class for creating VAR analysis plots."""
    
    def __init__(self, config: VARConfig):
        self.config = config
        self.ylims = {}  # Store y-axis limits for each variable
        
    def create_comparison_plot(self, 
                             cholesky_results: Tuple,
                             iv_results: Dict[str, Dict],
                             var_names_long: List[str],
                             output_path: Path = None) -> None:
        """Create comparison plot between Cholesky and IV identification.
        
        Parameters
        ----------
        cholesky_results : Tuple
            Tuple containing (INF, SUP, MED, BAR) for Cholesky identification
        iv_results : Dict[str, Dict]
            Dictionary containing results for each IV, with keys 'INF', 'SUP', 'MED', 'BAR'
        var_names_long : List[str]
            List of long variable names for plot titles
        output_path : Path, optional
            Path to save the plot. If None, uses './figures'
        """
        plt.figure(figsize=(20, 20))
        
        # First pass to determine y-axis limits
        self._compute_ylims(cholesky_results, iv_results)
        
        # Create plots
        for ii, var in enumerate(self.config.var_names):
            # Cholesky subplot (left column)
            ax1 = plt.subplot(4, 2, 2*ii + 1)
            self._plot_cholesky_subplot(cholesky_results, var, var_names_long[ii], ax1)
            
            # IV subplot (right column)
            ax2 = plt.subplot(4, 2, 2*ii + 2)
            self._plot_iv_subplot(iv_results, var, var_names_long[ii], ax2)
            
            # Set same y-axis limits for both plots
            ax1.set_ylim(self.ylims[var])
            ax2.set_ylim(self.ylims[var])
            
            # Add x-label only for bottom plots
            if ii == len(self.config.var_names) - 1:
                ax1.set_xlabel('Months')
                ax2.set_xlabel('Months')
        
        plt.tight_layout()
        self._save_plot('irf_gk2015.pdf', output_path)
    
    def _compute_ylims(self, cholesky_results, iv_results):
        """Compute common y-axis limits for each variable."""
        INF, SUP, MED, BAR = cholesky_results
        
        for var in self.config.var_names:
            # Get min/max from Cholesky
            mins = [INF.loc[:, var].min(), SUP.loc[:, var].min()]
            maxs = [INF.loc[:, var].max(), SUP.loc[:, var].max()]
            
            # Get min/max from IV
            for iv_var in self.config.iv_names:
                mins.extend([
                    iv_results['INF'][iv_var].loc[:, var].min(),
                    iv_results['SUP'][iv_var].loc[:, var].min()
                ])
                maxs.extend([
                    iv_results['INF'][iv_var].loc[:, var].max(),
                    iv_results['SUP'][iv_var].loc[:, var].max()
                ])
            
            # Add some padding
            ymin, ymax = min(mins), max(maxs)
            padding = (ymax - ymin) * 0.1
            self.ylims[var] = (ymin - padding, ymax + padding)
    
    def _setup_axis(self, ax, title: str):
        """Setup common axis properties."""
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(title, fontsize=16, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    def _plot_cholesky_subplot(self, cholesky_results, var: str, title: str, ax) -> None:
        """Plot Cholesky identification subplot."""
        INF, SUP, MED, BAR = cholesky_results
        # Plot median response
        ax.plot(BAR.loc[:, var], '-r', linewidth=2, label='Cholesky', zorder=3)
        
        # Plot confidence bands
        ax.fill_between(
            range(len(INF)), 
            INF.loc[:, var], 
            SUP.loc[:, var], 
            color='r', 
            alpha=0.1,
            zorder=1
        )
        
        # Plot band boundaries as dashed lines
        ax.plot(INF.loc[:, var], '--r', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(SUP.loc[:, var], '--r', linewidth=1, alpha=0.5, zorder=2)
        
        # Plot zero line
        ax.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5, zorder=0)
        
        self._setup_axis(ax, title)
        ax.legend()
    
    def _plot_iv_subplot(self, iv_results: Dict, var: str, title: str, ax) -> None:
        """Plot IV identification subplot."""
        colors = ['k', 'b', 'g']
        
        for i, iv_var in enumerate(self.config.iv_names):
            # Plot median response
            ax.plot(
                iv_results['MED'][iv_var].loc[:, var], 
                '-', 
                color=colors[i], 
                linewidth=2, 
                label=f'IV ({iv_var})',
                zorder=3
            )
            
            # Plot confidence bands
            ax.fill_between(
                range(len(iv_results['INF'][iv_var])),
                iv_results['INF'][iv_var].loc[:, var],
                iv_results['SUP'][iv_var].loc[:, var],
                color=colors[i],
                alpha=0.1,
                zorder=1
            )
            
            # Plot band boundaries as dashed lines
            ax.plot(
                iv_results['INF'][iv_var].loc[:, var], 
                '--', 
                color=colors[i], 
                linewidth=1, 
                alpha=0.5,
                zorder=2
            )
            ax.plot(
                iv_results['SUP'][iv_var].loc[:, var], 
                '--', 
                color=colors[i], 
                linewidth=1, 
                alpha=0.5,
                zorder=2
            )
        
        # Plot zero line
        ax.plot(np.zeros(self.config.nsteps), '-k', linewidth=0.5, zorder=0)
        
        self._setup_axis(ax, title)
        ax.legend()
    
    def _save_plot(self, filename: str, output_path: Path = None) -> None:
        """Save plot to file."""
        if output_path is None:
            output_path = Path('figures')
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{Fore.GREEN}Plot saved as {filename}{Style.RESET_ALL}")
