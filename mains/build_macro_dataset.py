"""
Build a consolidated macroeconomic dataset from FRED data, EBP, and commodity price index.

This script:
1. Reads all FRED data series from the specified folders
2. Integrates Excess Bond Premium (EBP) data
3. Integrates commodity price index data
4. Integrates real GDP data
5. Saves the consolidated dataset to CSV

Usage:
    poetry run python -m mains.build_macro_dataset
"""

import os
import sys
from pathlib import Path
import argparse
import pandas as pd

# Add the project root to the Python path if not already there
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.varkit.utils.macro_dataset import MacroDatasetBuilder


def get_series_ids_from_dir(fred_dir: Path) -> list[str]:
    """
    Get all FRED series IDs from the given directory.
    
    Parameters
    ----------
    fred_dir : Path
        Directory containing FRED data folders
        
    Returns
    -------
    list[str]
        List of FRED series IDs
    """
    return [d.name for d in fred_dir.iterdir() if d.is_dir()]


def main():
    """Main function to build the macroeconomic dataset."""
    parser = argparse.ArgumentParser(description='Build a consolidated macroeconomic dataset.')
    
    parser.add_argument(
        '--fred-dir', 
        type=str, 
        default='macroeconomics_data/data/fred',
        help='Directory containing FRED data folders'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/raw',
        help='Directory to save the output dataset'
    )
    
    parser.add_argument(
        '--ebp-path', 
        type=str, 
        default='data/raw/ebp_csv.csv',
        help='Path to the EBP data file'
    )
    
    parser.add_argument(
        '--commodity-path', 
        type=str, 
        default='data/raw/commodity_price_index.csv',
        help='Path to the commodity price index file'
    )
    
    parser.add_argument(
        '--gdp-path', 
        type=str, 
        default='data/raw/gdp.csv',
        help='Path to the GDP data file'
    )
    
    parser.add_argument(
        '--output-filename', 
        type=str, 
        default='macrodata.csv',
        help='Name of the output dataset file'
    )
    
    parser.add_argument(
        '--metadata-filename', 
        type=str, 
        default='macrodata_metadata.txt',
        help='Name of the metadata report file'
    )
    
    args = parser.parse_args()
    
    # Convert path strings to Path objects
    fred_dir = Path(args.fred_dir)
    output_dir = Path(args.output_dir)
    ebp_path = Path(args.ebp_path) if args.ebp_path else None
    commodity_path = Path(args.commodity_path) if args.commodity_path else None
    gdp_path = Path(args.gdp_path) if args.gdp_path else None
    
    # Get all FRED series IDs from the directory
    series_ids = get_series_ids_from_dir(fred_dir)
    
    if not series_ids:
        print(f"No FRED data folders found in {fred_dir}")
        return
    
    print(f"Found {len(series_ids)} FRED series: {', '.join(series_ids)}")
    
    # Initialize the dataset builder
    builder = MacroDatasetBuilder(
        fred_dir=fred_dir,
        output_dir=output_dir,
        ebp_path=ebp_path,
        commodity_path=commodity_path,
        gdp_path=gdp_path
    )
    
    # Build the dataset
    print(f"Building dataset from {len(series_ids)} FRED series, EBP data, commodity price index, and GDP data...")
    dataset = builder.build_dataset(series_ids)
    
    # Print dataset info
    print(f"Dataset shape: {dataset.shape}")
    print(f"Date range: {dataset.index.min()} to {dataset.index.max()}")
    print(f"Variables: {', '.join(dataset.columns)}")
    
    # Save the dataset
    output_path = builder.save_dataset(dataset, args.output_filename)
    print(f"Dataset saved to {output_path}")
    
    # Generate metadata report
    metadata_path = builder.generate_metadata_report(args.metadata_filename)
    print(f"Metadata report saved to {metadata_path}")


if __name__ == "__main__":
    main() 