"""
Moving statistics functions.

This module implements various moving statistics corresponding to the MATLAB functions:
- MovAvg.m
- MovCorr.m
- MovStd.m
and others.
"""

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd


def moving_average(data: Union[np.ndarray, pd.DataFrame],
                  window: int,
                  center: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """Compute moving average.
    
    Args:
        data: Input array/DataFrame (nobs x nvars)
        window: Window size for moving average
        center: If True, window is centered on each point (default=False)
        
    Returns:
        Moving averages with same shape as input
        First (window-1) observations are NaN if not centered
    """
    # Input validation
    if window <= 0 or not isinstance(window, int):
        raise ValueError("Window must be a positive integer")
    
    # Convert to numpy if needed
    is_pandas = isinstance(data, pd.DataFrame)
    x = data.values if is_pandas else np.asarray(data)
    
    # Handle 1D arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    nobs, nvars = x.shape
    if window > nobs:
        raise ValueError("Window must not be greater than the length of data")
    
    # Compute moving average
    z = np.full_like(x, np.nan)
    
    if center:
        # Centered moving average
        radius = (window - 1) // 2
        for t in range(radius, nobs - radius):
            z[t] = np.nanmean(x[t-radius:t+radius+1], axis=0)
    else:
        # Forward moving average
        for t in range(window-1, nobs):
            z[t] = np.nanmean(x[t-window+1:t+1], axis=0)
    
    # Convert back to pandas if needed
    if is_pandas:
        z = pd.DataFrame(z, index=data.index, columns=data.columns)
    
    return z


def moving_correlation(data1: Union[np.ndarray, pd.DataFrame],
                      data2: Union[np.ndarray, pd.DataFrame],
                      window: int,
                      center: bool = False) -> Union[np.ndarray, pd.DataFrame]:
    """Compute moving correlation between two series.
    
    Args:
        data1: First input array/DataFrame (nobs x nvars)
        data2: Second input array/DataFrame (nobs x nvars)
        window: Window size for moving correlation
        center: If True, window is centered on each point (default=False)
        
    Returns:
        Moving correlations with same shape as input
        First (window-1) observations are NaN if not centered
    """
    # Input validation
    if window <= 0 or not isinstance(window, int):
        raise ValueError("Window must be a positive integer")
    
    # Convert to numpy if needed
    is_pandas = isinstance(data1, pd.DataFrame)
    x1 = data1.values if is_pandas else np.asarray(data1)
    x2 = data2.values if is_pandas else np.asarray(data2)
    
    # Handle 1D arrays
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)
    
    # Check dimensions
    if x1.shape != x2.shape:
        raise ValueError("Input arrays must have the same shape")
    
    nobs, nvars = x1.shape
    if window > nobs:
        raise ValueError("Window must not be greater than the length of data")
    
    # Compute moving correlation
    z = np.full_like(x1, np.nan)
    
    if center:
        # Centered moving correlation
        radius = (window - 1) // 2
        for t in range(radius, nobs - radius):
            for v in range(nvars):
                z[t, v] = np.corrcoef(
                    x1[t-radius:t+radius+1, v],
                    x2[t-radius:t+radius+1, v]
                )[0, 1]
    else:
        # Forward moving correlation
        for t in range(window-1, nobs):
            for v in range(nvars):
                z[t, v] = np.corrcoef(
                    x1[t-window+1:t+1, v],
                    x2[t-window+1:t+1, v]
                )[0, 1]
    
    # Convert back to pandas if needed
    if is_pandas:
        z = pd.DataFrame(z, index=data1.index, columns=data1.columns)
    
    return z


def moving_statistics(data: Union[np.ndarray, pd.DataFrame],
                     window: int,
                     center: bool = False) -> dict:
    """Compute multiple moving statistics.
    
    Args:
        data: Input array/DataFrame (nobs x nvars)
        window: Window size for moving statistics
        center: If True, window is centered on each point (default=False)
        
    Returns:
        Dictionary containing:
            - mean: Moving average
            - std: Moving standard deviation
            - min: Moving minimum
            - max: Moving maximum
    """
    # Input validation
    if window <= 0 or not isinstance(window, int):
        raise ValueError("Window must be a positive integer")
    
    # Convert to numpy if needed
    is_pandas = isinstance(data, pd.DataFrame)
    x = data.values if is_pandas else np.asarray(data)
    
    # Handle 1D arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    nobs, nvars = x.shape
    if window > nobs:
        raise ValueError("Window must not be greater than the length of data")
    
    # Initialize output arrays
    mean = np.full_like(x, np.nan)
    std = np.full_like(x, np.nan)
    min_val = np.full_like(x, np.nan)
    max_val = np.full_like(x, np.nan)
    
    if center:
        # Centered statistics
        radius = (window - 1) // 2
        for t in range(radius, nobs - radius):
            window_data = x[t-radius:t+radius+1]
            mean[t] = np.nanmean(window_data, axis=0)
            std[t] = np.nanstd(window_data, axis=0)
            min_val[t] = np.nanmin(window_data, axis=0)
            max_val[t] = np.nanmax(window_data, axis=0)
    else:
        # Forward statistics
        for t in range(window-1, nobs):
            window_data = x[t-window+1:t+1]
            mean[t] = np.nanmean(window_data, axis=0)
            std[t] = np.nanstd(window_data, axis=0)
            min_val[t] = np.nanmin(window_data, axis=0)
            max_val[t] = np.nanmax(window_data, axis=0)
    
    # Convert back to pandas if needed
    if is_pandas:
        mean = pd.DataFrame(mean, index=data.index, columns=data.columns)
        std = pd.DataFrame(std, index=data.index, columns=data.columns)
        min_val = pd.DataFrame(min_val, index=data.index, columns=data.columns)
        max_val = pd.DataFrame(max_val, index=data.index, columns=data.columns)
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    } 