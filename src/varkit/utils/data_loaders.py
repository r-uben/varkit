"""
Data loaders for various macroeconomic data sources.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import pandas as pd
import calendar
from datetime import datetime


@dataclass
class FredDataInfo:
    """Information about a FRED data series."""
    series_id: str
    title: str
    units: str
    frequency: str
    observation_start: str
    observation_end: str
    notes: str


class DateFormatHandler:
    """Handle different date formats in data files."""
    
    @staticmethod
    def parse_fred_date(date_str: str) -> pd.Timestamp:
        """Parse date string from FRED data file."""
        date = pd.to_datetime(date_str)
        return DateFormatHandler.set_to_end_of_period(date)
    
    @staticmethod
    def parse_commodity_date(date_str: str) -> pd.Timestamp:
        """Parse date string from commodity price index file."""
        date = pd.to_datetime(date_str, format="%b %Y")
        return DateFormatHandler.set_to_end_of_period(date)
    
    @staticmethod
    def parse_ebp_date(date_str: str) -> pd.Timestamp:
        """Parse date string from EBP file."""
        date = pd.to_datetime(date_str)
        return DateFormatHandler.set_to_end_of_period(date)
    
    @staticmethod
    def set_to_end_of_period(date: pd.Timestamp) -> pd.Timestamp:
        """Set date to the last day of the month."""
        last_day = calendar.monthrange(date.year, date.month)[1]
        return pd.Timestamp(date.year, date.month, last_day)
    
    @staticmethod
    def convert_to_monthly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Convert a dataframe with dates to monthly frequency."""
        # Ensure the date column is a datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Convert all dates to end-of-month dates before setting index
        df[date_col] = df[date_col].apply(DateFormatHandler.set_to_end_of_period)
        
        # Set the date column as index
        df.set_index(date_col, inplace=True)
        
        # Determine the original frequency
        if df.index.inferred_freq == 'QS' or df.index.inferred_freq == 'Q' or df.index.inferred_freq == 'Q-DEC':
            # Quarterly data - for each quarter, use the last month of the quarter (Mar, Jun, Sep, Dec)
            # and set to the last day of those months
            df = df.resample('QS').last()  # This gets the first day of each quarter
            # Convert quarterly dates to end-of-quarter dates
            new_index = df.index.map(lambda d: pd.Timestamp(d.year, 3 * ((d.month - 1) // 3 + 1), 
                                                         calendar.monthrange(d.year, 3 * ((d.month - 1) // 3 + 1))[1]))
            df.index = new_index
            
            # Now convert to monthly with forward fill, but only for the months in the quarter
            monthly_df = df.asfreq('ME', method='ffill')  # End of month frequency
            
            # Only keep the last month of each quarter
            quarterly_months = {3, 6, 9, 12}
            monthly_df = monthly_df[monthly_df.index.month.isin(quarterly_months)]
            
            # Reindex to all months and forward fill within quarters
            all_months = pd.date_range(start=monthly_df.index.min(), end=monthly_df.index.max(), freq='ME')
            df = monthly_df.reindex(all_months).ffill()
            
        elif df.index.inferred_freq == 'D' or df.index.inferred_freq == 'B':
            # Daily data - take the last value of each month, ensuring it's the last day
            df = df.resample('ME').last()  # End of month resampling
        else:
            # For monthly data, ensure all dates are end of month
            df = df.resample('ME').last()
            
        return df


class FredDataLoader:
    """Load data from FRED data folders."""
    
    @staticmethod
    def read_metadata(metadata_path: Path) -> FredDataInfo:
        """Read metadata file from FRED data folder."""
        metadata_dict = {}
        with open(metadata_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metadata_dict[key.strip()] = value.strip()
                
        return FredDataInfo(
            series_id=metadata_dict.get('series_id', ''),
            title=metadata_dict.get('title', ''),
            units=metadata_dict.get('units', ''),
            frequency=metadata_dict.get('frequency', ''),
            observation_start=metadata_dict.get('observation_start', ''),
            observation_end=metadata_dict.get('observation_end', ''),
            notes=metadata_dict.get('notes', '')
        )
    
    @staticmethod
    def read_data(data_path: Path) -> pd.DataFrame:
        """Read data file from FRED data folder."""
        return pd.read_csv(data_path)
    
    @staticmethod
    def load_fred_series(fred_dir: Path, series_id: str) -> tuple[pd.DataFrame, FredDataInfo]:
        """Load a FRED data series from a directory."""
        series_dir = fred_dir / series_id
        metadata_path = series_dir / "metadata.txt"
        data_path = series_dir / "data.csv"
        
        metadata = FredDataLoader.read_metadata(metadata_path)
        data = FredDataLoader.read_data(data_path)
        
        # Convert date column to timestamp and set to end of period
        data['date'] = data['date'].apply(DateFormatHandler.parse_fred_date)
        
        return data, metadata


class EBPDataLoader:
    """Load Excess Bond Premium data."""
    
    @staticmethod
    def load_data(data_path: Path) -> pd.DataFrame:
        """Load EBP data from CSV file."""
        df = pd.read_csv(data_path)
        df['date'] = df['date'].apply(DateFormatHandler.parse_ebp_date)
        return df


class CommodityPriceLoader:
    """Load commodity price index data."""
    
    @staticmethod
    def load_data(data_path: Path) -> pd.DataFrame:
        """Load commodity price index data from CSV file."""
        df = pd.read_csv(data_path)
        df['date'] = df['Month'].apply(DateFormatHandler.parse_commodity_date)
        df.drop('Month', axis=1, inplace=True)
        
        # Convert the 'Change' column which contains percentages as strings
        df['Change'] = df['Change'].replace('-', '0%')
        df['Change'] = df['Change'].str.rstrip('%').astype('float') / 100.0
        
        return df 


class GDPDataLoader:
    """Load GDP data from CSV file."""
    
    @staticmethod
    def load_data(data_path: Path) -> pd.DataFrame:
        """Load GDP data from CSV file.
        
        Args:
            data_path: Path to the GDP data CSV file.
            
        Returns:
            DataFrame containing GDP data with date index and GDP components.
        """
        df = pd.read_csv(data_path)
        
        # Convert Year and Month to datetime
        df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
        df = df.drop(['Year', 'Month'], axis=1)
        
        # Set date as index and ensure it's end of month
        df['date'] = df['date'].apply(DateFormatHandler.set_to_end_of_period)
        df.set_index('date', inplace=True)
        
        return df 