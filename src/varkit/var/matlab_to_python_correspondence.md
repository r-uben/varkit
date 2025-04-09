# MATLAB to Python Correspondence

This document maps the original MATLAB functions from the VAR-Toolbox to their Python implementations.

## Core VAR Modeling

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| VARmodel.m | var_model.py | Main VAR model estimation |
| VARlag.m | var_lag.py | Lag selection criteria |
| VARmakelags.m | var_make_lags.py | Create lagged variables |
| VARmakexy.m | var_make_xy.py | Prepare X and Y matrices |
| VARoption.m | var_options.py | Model configuration options |

## Impulse Response Analysis

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| VARir.m | var_impulse_response.py | Impulse response functions |
| VARirband.m | var_ir_bands.py | Confidence bands for IRFs |
| VARirplot.m | var_ir_plot.py | Plot impulse responses |

## Variance Decomposition

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| VARvd.m | var_variance_decomp.py | Variance decomposition |
| VARvdband.m | var_vd_bands.py | Confidence bands for VD |
| VARvdplot.m | var_vd_plot.py | Plot variance decomposition |

## Historical Decomposition

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| VARhd.m | var_historical_decomp.py | Historical decomposition |
| VARhdplot.m | var_hd_plot.py | Plot historical decomposition |

## ARDL and OLS Models

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| ARDLmodel.m | ardl_model.py | ARDL model estimation |
| ARDLprint.m | ardl_results.py | Print ARDL results |
| OLSmodel.m | ols_model.py | OLS model estimation |
| OLSprint.m | ols_results.py | Print OLS results |

## Sign Restrictions

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| SR.m | sign_restrictions.py | Sign restrictions implementation |
| SignRestrictions.m | sign_restrictions_utils.py | Sign restrictions utilities |
| SRhdplot.m | sr_hd_plot.py | Plot SR historical decomposition |
| SRirplot.m | sr_ir_plot.py | Plot SR impulse responses |
| SRvdplot.m | sr_vd_plot.py | Plot SR variance decomposition |

## Posterior Analysis

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| VARdrawpost.m | var_posterior.py | Draw from posterior distribution |

## Utilities

| MATLAB File | Python File | Description |
|------------|-------------|-------------|
| L.m | lag_operator.py | Lag operator utilities |
| OrthNorm.m | orthogonal_normalization.py | Orthogonal normalization |
| VARprint.m | var_results.py | Print VAR results |

## Implementation Notes

1. All Python implementations will follow object-oriented programming principles
2. Visualization will be implemented using matplotlib/plotly
3. Core numerical operations will use numpy and scipy
4. Statistical operations will leverage statsmodels where appropriate
5. All classes will include proper type hints and docstrings
6. Unit tests will be provided for each module

## Dependencies

- numpy
- scipy
- pandas
- matplotlib/plotly
- statsmodels 