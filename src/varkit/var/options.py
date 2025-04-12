"""
VAR model options.

This module implements the options for VAR models, corresponding to the original VARoption.m
"""

class Options:
    """Optional inputs for VAR analysis."""

    def __init__(self):
        """Initialize the VAR options with default values."""
        self.vnames = None      # endogenous variables names
        self.vnames_ex = None   # exogenous variables names
        self.snames = None      # shocks names
        self.nsteps = 40        # number of steps for computation of IRFs and FEVDs
        self.impact = 0         # size of the shock for IRFs: 0=1stdev, 1=unit shock
        self.shut = 0           # forces the IRF of one variable to zero
        self.ident = 'short'    # identification method for IRFs
        self.recurs = 'wold'    # method for computation of recursive stuff
        self.ndraws = 1000      # number of draws for bootstrap or sign restrictions
        self.mult = 10          # multiple of draws to be printed at screen
        self.pctg = 95          # confidence level for bootstrap
        self.method = 'bs'      # methodology for error bands
        self.sr_hor = 1         # number of periods that sign restrictions are imposed on
        self.sr_rot = 500       # max number of rotations for finding sign restrictions
        self.sr_draw = 100000   # max number of total draws for finding sign restrictions
        self.sr_mod = 1         # model uncertainty for sign restrictions
        self.pick = 0           # selects one variable for IRFs and FEVDs plots
        self.quality = 2        # quality of exported figures
        self.suptitle = 0       # title on top of figures
        self.datesnum = None    # numeric vector of dates in the VAR
        self.datestxt = None     # cell vector of dates in the VAR
        self.datestype = 1      # 1 smart labels; 2 less smart labels
        self.firstdate = None   # initial date of the sample
        self.frequency = 'q'    # frequency of the data
        self.figname = None     # string for name of exported figure
        self.FigSize = [26, 24]  # size of window for plots

    def to_dict(self) -> dict:
        """Convert the options to a dictionary."""
        return {
            'vnames': self.vnames,
            'vnames_ex': self.vnames_ex,
            'snames': self.snames,
            'nsteps': self.nsteps,
            'impact': self.impact,
            'shut': self.shut,
            'ident': self.ident,
            'recurs': self.recurs,
            'ndraws': self.ndraws,
            'mult': self.mult,
            'pctg': self.pctg,
            'method': self.method,
            'sr_hor': self.sr_hor,
            'sr_rot': self.sr_rot,
            'sr_draw': self.sr_draw,
            'sr_mod': self.sr_mod,
            'pick': self.pick,
            'quality': self.quality,
            'suptitle': self.suptitle,
            'datesnum': self.datesnum,
            'datestxt': self.datestxt,
            'datestype': self.datestype,
            'firstdate': self.firstdate,
            'frequency': self.frequency,
            'figname': self.figname,
            'FigSize': self.FigSize
        }