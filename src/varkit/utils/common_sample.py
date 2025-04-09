"""Common sample functions ported from MATLAB's VAR-Toolbox Utils folder."""

import numpy as np

def common_sample(data, dim=0):
    """If a row of data contains a NaN, the row is removed.
    
    This function exactly matches MATLAB's CommonSample.m behavior, which removes rows 
    or columns with NaNs from a data matrix. This is crucial for aligning time series 
    data for IV identification.
    
    Args:
        data: Numpy array or matrix of data
        dim: Dimension to operate on (0=rows, 1=columns)
    
    Returns:
        tuple:
            - OUT: Common sample matrix without NaNs
            - fo: Number of NaNs at beginning
            - lo: Number of NaNs at end
            
    Example:
        >>> x = np.array([[1, 2], [np.nan, 4], [5, 6]])
        >>> out, fo, lo = common_sample(x)
        >>> print(out)  # Should be [[1, 2], [5, 6]]
    """
    # Initialize counters for NaNs at beginning and end
    fo = 0
    lo = 0
    
    # Make a copy to avoid modifying the original
    data_copy = data.copy()
    
    if dim == 0:  # Process rows
        # Sum along rows to detect NaNs (NaN + any value = NaN)
        temp = np.sum(data, axis=1)
        
        # Count NaNs at the beginning
        ii = 0
        if ii < len(temp) and np.isnan(temp[ii]):
            while ii < len(temp) and np.isnan(temp[ii]):
                fo += 1
                ii += 1
        
        # Count NaNs at the end
        for ii in range(len(temp) - fo):
            if np.isnan(temp[len(temp) - 1 - ii]):
                lo += 1
            else:
                break  # Stop at first non-NaN
        
        # Remove rows with any NaN
        # This is the key part that matches MATLAB's behavior
        data_out = data_copy[~np.any(np.isnan(data_copy), axis=1)]
    
    else:  # Process columns (dim=1)
        # Sum along columns to detect NaNs
        temp = np.sum(data, axis=0)
        
        # Count NaNs at the beginning
        ii = 0
        if ii < len(temp) and np.isnan(temp[ii]):
            while ii < len(temp) and np.isnan(temp[ii]):
                fo += 1
                ii += 1
        
        # Count NaNs at the end
        for ii in range(len(temp) - fo):
            if np.isnan(temp[len(temp) - 1 - ii]):
                lo += 1
            else:
                break  # Stop at first non-NaN
        
        # Remove columns with any NaN
        data_out = data_copy[:, ~np.any(np.isnan(data_copy), axis=0)]
    
    return data_out, fo, lo 