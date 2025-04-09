"""
VAR (Vector Autoregression) module for time series analysis.

This module provides a comprehensive implementation of Vector Autoregression analysis,
including estimation, inference, and various decomposition methods.
"""

from .var_model import VARModel, VAROptions, VARResults
from .var_ir import var_ir
from .var_irband import var_irband

__all__ = ['VARModel', 'VAROptions', 'VARResults', 'var_ir', 'var_irband'] 