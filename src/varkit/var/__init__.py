"""
VAR (Vector Autoregression) module for time series analysis.

This module provides a comprehensive implementation of Vector Autoregression analysis,
including estimation, inference, and various decomposition methods.
"""

from .model import Model, Options, Output
from .impulse_response import ImpulseResponse

__all__ = ['Model', 'Options', 'Output', 'ImpulseResponse'] 