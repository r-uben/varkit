"""
Utilities for building and managing macroeconomic datasets from multiple sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

from .data_loaders import (
    FredDataLoader, 
    EBPDataLoader, 
    CommodityPriceLoader,
    GDPDataLoader,
    DateFormatHandler,
    FredDataInfo
)


class MacroDatasetBuilder:
    """Class to build a consolidated macroeconomic dataset from multiple sources."""
    
    def __init__(
        self, 
        fred_dir: Path, 
        output_dir: Path,
        ebp_path: Optional[Path] = None,
        commodity_path: Optional[Path] = None,
        gdp_path: Optional[Path] = None
    ):
        """
        Initialize the dataset builder.
        
        Parameters
        ----------
        fred_dir : Path
            Directory containing FRED data folders
        output_dir : Path
            Directory to save the output dataset
        ebp_path : Optional[Path]
            Path to the EBP data file
        commodity_path : Optional[Path]
            Path to the commodity price index file
        gdp_path : Optional[Path]
            Path to the GDP data file
        """
        self.fred_dir = fred_dir
        self.output_dir = output_dir
        self.ebp_path = ebp_path
        self.commodity_path = commodity_path
        self.gdp_path = gdp_path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to store metadata for each series
        self.metadata: Dict[str, FredDataInfo] = {}
        
    def load_fred_series(self, series_ids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load FRED data series.
        
        Parameters
        ----------
        series_ids : List[str]
            List of FRED series IDs to load
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping series IDs to dataframes
        """
        series_data = {}
        
        for series_id in series_ids:
            try:
                data, metadata = FredDataLoader.load_fred_series(self.fred_dir, series_id)
                
                # Store metadata
                self.metadata[series_id] = metadata
                
                # Convert to monthly frequency if needed
                if metadata.frequency.lower() != 'monthly':
                    data = DateFormatHandler.convert_to_monthly(data)
                else:
                    # Set date as index for consistency and ensure end-of-month dates
                    data.set_index('date', inplace=True)
                    data = data.resample('ME').last()  # Ensure end-of-month even for monthly data
                
                # Rename the value column to the series ID
                data.rename(columns={'value': series_id}, inplace=True)
                
                series_data[series_id] = data
                
            except Exception as e:
                print(f"Error loading FRED series {series_id}: {e}")
        
        return series_data
    
    def load_ebp_data(self) -> Optional[pd.DataFrame]:
        """
        Load Excess Bond Premium (EBP) data.
        
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing EBP data, or None if no path is provided
        """
        if self.ebp_path is None:
            return None
        
        try:
            data = EBPDataLoader.load_data(self.ebp_path)
            data.set_index('date', inplace=True)
            # Ensure end-of-month dates
            data = data.resample('ME').last()
            
            return data
        except Exception as e:
            print(f"Error loading EBP data: {e}")
            return None
    
    def load_commodity_data(self) -> Optional[pd.DataFrame]:
        """
        Load commodity price index data.
        
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing commodity price data, or None if no path is provided
        """
        if self.commodity_path is None:
            return None
        
        try:
            data = CommodityPriceLoader.load_data(self.commodity_path)
            data.set_index('date', inplace=True)
            # Ensure end-of-month dates
            data = data.resample('ME').last()
            
            # Rename Price column to commodity_price for clarity
            data.rename(columns={'Price': 'commodity_price'}, inplace=True)
            
            return data
        except Exception as e:
            print(f"Error loading commodity price data: {e}")
            return None
            
    def load_gdp_data(self) -> Optional[pd.DataFrame]:
        """
        Load GDP data.
        
        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame containing GDP data, or None if no path is provided
        """
        if self.gdp_path is None:
            return None
            
        try:
            data = GDPDataLoader.load_data(self.gdp_path)
            # Extract only the GDP_real column and rename it for clarity
            gdp_data = pd.DataFrame({'gdp_real': data['GDP_real']}, index=data.index)
            return gdp_data
        except Exception as e:
            print(f"Error loading GDP data: {e}")
            return None
    
    def build_dataset(self, series_ids: List[str]) -> pd.DataFrame:
        """
        Build a consolidated dataset from multiple sources.
        
        Parameters
        ----------
        series_ids : List[str]
            List of FRED series IDs to include
            
        Returns
        -------
        pd.DataFrame
            Consolidated dataset with all variables
        """
        # Load FRED series
        fred_data = self.load_fred_series(series_ids)
        
        # Start with an empty dataframe
        dataset = pd.DataFrame()
        
        # Merge FRED data
        for series_id, data in fred_data.items():
            if dataset.empty:
                dataset = data
            else:
                dataset = dataset.join(data, how='outer')
        
        # Load and merge EBP data if available
        ebp_data = self.load_ebp_data()
        if ebp_data is not None:
            dataset = dataset.join(ebp_data, how='outer')
        
        # Load and merge commodity price data if available
        commodity_data = self.load_commodity_data()
        if commodity_data is not None:
            dataset = dataset.join(commodity_data, how='outer')
            
        # Load and merge GDP data if available
        gdp_data = self.load_gdp_data()
        if gdp_data is not None:
            dataset = dataset.join(gdp_data, how='outer')
        
        # Sort by date
        dataset.sort_index(inplace=True)
        
        # Final check to ensure all dates are end-of-month
        dataset = dataset.resample('ME').last()
        
        return dataset
    
    def save_dataset(self, dataset: pd.DataFrame, filename: str = "macrodata.csv") -> Path:
        """
        Save the dataset to a CSV file.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset to save
        filename : str
            Name of the output file
            
        Returns
        -------
        Path
            Path to the saved file
        """
        # Reset index to make date a column
        dataset_out = dataset.reset_index()
        
        # Rename index column to 'date' for clarity
        dataset_out.rename(columns={'index': 'date'}, inplace=True)
        
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = self.output_dir / filename
        dataset_out.to_csv(output_path, index=False)
        
        return output_path
    
    def generate_metadata_report(self, output_filename: str = "metadata_report.txt") -> Path:
        """
        Generate a report of metadata for all loaded series.
        
        Parameters
        ----------
        output_filename : str
            Name of the output file
            
        Returns
        -------
        Path
            Path to the saved report
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w') as f:
            f.write("# Macroeconomic Dataset Metadata\n\n")
            f.write("All dates in the dataset are set to the last day of each month.\n\n")
            
            # FRED series metadata
            f.write("## FRED Series\n\n")
            for series_id, metadata in self.metadata.items():
                f.write(f"### {series_id}: {metadata.title}\n")
                f.write(f"- Units: {metadata.units}\n")
                f.write(f"- Original Frequency: {metadata.frequency}\n")
                f.write(f"- Observation range: {metadata.observation_start} to {metadata.observation_end}\n")
                f.write(f"- Notes: {metadata.notes[:200]}...\n\n")
            
            # EBP data
            if self.ebp_path is not None:
                f.write("## Excess Bond Premium (EBP)\n\n")
                f.write(f"- Source: {self.ebp_path}\n")
                f.write("- Variables: gz_spread, ebp, est_prob\n\n")
            
            # Commodity price data
            if self.commodity_path is not None:
                f.write("## Commodity Price Index\n\n")
                f.write(f"- Source: {self.commodity_path}\n")
                f.write("- Variables: commodity_price, Change\n\n")
        
        return output_path 