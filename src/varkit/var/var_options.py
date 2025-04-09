"""
VAR model options.

This module implements the options for VAR models, corresponding to the original VARoption.m
"""

def var_option() -> dict:
    """Optional inputs for VAR analysis.
    
    This function is run automatically in the var_model function.
    
    Returns:
        Dictionary with VAR options
    """
    return {
        'vnames': None,      # endogenous variables names
        'vnames_ex': None,   # exogenous variables names
        'snames': None,      # shocks names
        'nsteps': 40,        # number of steps for computation of IRFs and FEVDs
        'impact': 0,         # size of the shock for IRFs: 0=1stdev, 1=unit shock
        'shut': 0,           # forces the IRF of one variable to zero
        'ident': 'short',    # identification method for IRFs ('short' zero short-run restr, 'long' zero long-run restr, 'sign' sign restr, 'iv' external instrument)
        'recurs': 'wold',    # method for computation of recursive stuff ('wold' form MA representation, 'comp' for companion form)
        'ndraws': 1000,      # number of draws for bootstrap or sign restrictions
        'mult': 10,          # multiple of draws to be printed at screen
        'pctg': 95,          # confidence level for bootstrap
        'method': 'bs',      # methodology for error bands, 'bs' for standard bootstrap, 'wild' wild bootstrap
        'sr_hor': 1,         # number of periods that sign restrictions are imposed on
        'sr_rot': 500,       # max number of rotations for finding sign restrictions
        'sr_draw': 100000,   # max number of total draws for finding sign restrictions
        'sr_mod': 1,         # model uncertainty for sign restrictions (1=yes, 0=no)
        'pick': 0,           # selects one variable for IRFs and FEVDs plots (0 => plot all)
        'quality': 2,        # quality of exported figures: 2=high (exportgraphics), 1=high (ghostscript), 0=low
        'suptitle': 0,       # title on top of figures
        'datesnum': None,    # numeric vector of dates in the VAR
        'datestxt': None,    # cell vector of dates in the VAR
        'datestype': 1,      # 1 smart labels; 2 less smart labels
        'firstdate': None,   # initial date of the sample in format 1999.75 => 1999Q4 (both for annual and quarterly data)
        'frequency': 'q',    # frequency of the data: 'm' monthly, 'q' quarterly, 'y' yearly
        'figname': None,     # string for name of exported figure
        'FigSize': [26, 24]  # size of window for plots
    } 