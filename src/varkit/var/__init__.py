"""
VAR (Vector Autoregression) module for time series analysis.

This module provides a comprehensive implementation of Vector Autoregression analysis,
including estimation, inference, and various decomposition methods.
"""

from .var_model import VARModel, VAROptions

__all__ = ['VARModel', 'VAROptions'] 