"""
Data handling utilities.

This module implements various data handling utilities corresponding to the MATLAB functions:
- CommonSample.m
- NaN2Num.m
- Num2NaN.m
- datatreat.m
and others.
"""

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd


def common_sample(*args: Union[np.ndarray, pd.DataFrame]) -> tuple:
    """Extract common sample from multiple arrays/DataFrames.
    
    Args:
        *args: Arrays/DataFrames to find common sample
        
    Returns:
        Tuple of arrays/DataFrames with common sample
    """
    # Convert all inputs to pandas
    dfs = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            if arg.ndim == 1:
                df = pd.DataFrame(arg.reshape(-1, 1))
            else:
                df = pd.DataFrame(arg)
        else:
            df = arg.copy()
        dfs.append(df)
    
    # Find common non-NaN observations
    mask = np.ones(len(dfs[0]), dtype=bool)
    for df in dfs:
        mask &= ~df.isna().any(axis=1)
    
    # Apply mask
    result = []
    for df in dfs:
        filtered = df[mask]
        result.append(filtered.values if isinstance(args[0], np.ndarray) else filtered)
    
    return tuple(result)


def nan_to_num(data: Union[np.ndarray, pd.DataFrame],
               value: float = 0.0) -> Union[np.ndarray, pd.DataFrame]:
    """Replace NaN values with a specified number.
    
    Args:
        data: Input array/DataFrame
        value: Value to replace NaNs with (default=0.0)
        
    Returns:
        Array/DataFrame with NaNs replaced
    """
    if isinstance(data, pd.DataFrame):
        return data.fillna(value)
    else:
        return np.nan_to_num(data, nan=value)


def num_to_nan(data: Union[np.ndarray, pd.DataFrame],
               value: float = 0.0) -> Union[np.ndarray, pd.DataFrame]:
    """Replace specified number with NaN.
    
    Args:
        data: Input array/DataFrame
        value: Value to replace with NaN (default=0.0)
        
    Returns:
        Array/DataFrame with specified values replaced by NaN
    """
    if isinstance(data, pd.DataFrame):
        return data.replace(value, np.nan)
    else:
        return np.where(data == value, np.nan, data)


def winsorize(data: Union[np.ndarray, pd.DataFrame],
              limits: Union[float, Tuple[float, float]] = 0.05,
              axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """Winsorize data to limit extreme values.
    
    Args:
        data: Input array/DataFrame
        limits: Proportion to cut on each tail (default=0.05)
               If tuple, (lower, upper) proportions
        axis: Axis along which to winsorize (default=0)
        
    Returns:
        Winsorized array/DataFrame
    """
    # Convert to numpy array
    is_pandas = isinstance(data, pd.DataFrame)
    x = data.values if is_pandas else np.asarray(data)
    
    # Handle 1D arrays
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Convert limits to tuple if scalar
    if isinstance(limits, (int, float)):
        limits = (limits, limits)
    
    # Compute percentiles
    lower = np.nanpercentile(x, limits[0] * 100, axis=axis)
    upper = np.nanpercentile(x, (1 - limits[1]) * 100, axis=axis)
    
    # Reshape for broadcasting
    if axis == 0:
        lower = np.broadcast_to(lower, (x.shape[0], 1))
        upper = np.broadcast_to(upper, (x.shape[0], 1))
    else:
        lower = np.broadcast_to(lower, (1, x.shape[1]))
        upper = np.broadcast_to(upper, (1, x.shape[1]))
    
    # Winsorize
    x = np.maximum(np.minimum(x, upper), lower)
    
    # Convert back to pandas if needed
    if is_pandas:
        x = pd.DataFrame(x, index=data.index, columns=data.columns)
    
    return x


def table_print(data: Union[np.ndarray, pd.DataFrame],
                row_names: Optional[list] = None,
                col_names: Optional[list] = None,
                precision: int = 4,
                title: Optional[str] = None) -> str:
    """Create formatted string table from data.
    
    Args:
        data: Input array/DataFrame
        row_names: Optional list of row names
        col_names: Optional list of column names
        precision: Number of decimal places (default=4)
        title: Optional table title
        
    Returns:
        Formatted string table
    """
    # Convert to numpy array
    x = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
    
    # Get dimensions
    nrows, ncols = x.shape
    
    # Get names
    if isinstance(data, pd.DataFrame):
        row_names = row_names or data.index.astype(str).tolist()
        col_names = col_names or data.columns.astype(str).tolist()
    else:
        row_names = row_names or [f"Row{i+1}" for i in range(nrows)]
        col_names = col_names or [f"Col{i+1}" for i in range(ncols)]
    
    # Format number format
    fmt = f"{{:.{precision}f}}"
    
    # Calculate column widths
    col_widths = [max(len(str(col)), len(fmt.format(np.nanmax(abs(x[:, i])))))
                  for i, col in enumerate(col_names)]
    row_width = max(len(str(row)) for row in row_names)
    
    # Create header
    if title:
        table = [title]
    else:
        table = []
    
    # Add column names
    header = " " * row_width + " | "
    header += " | ".join(f"{col:>{width}}" for col, width in zip(col_names, col_widths))
    table.append(header)
    
    # Add separator
    separator = "-" * row_width + "-+-" + "-+-".join("-" * width for width in col_widths)
    table.append(separator)
    
    # Add data rows
    for i, row_name in enumerate(row_names):
        row = f"{row_name:>{row_width}} | "
        row += " | ".join(f"{fmt.format(x[i,j]):>{width}}" 
                         for j, width in enumerate(col_widths))
        table.append(row)
    
    return "\n".join(table) 