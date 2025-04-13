import pandas as pd

class GeneralUtils:

    @staticmethod
    def get_common_sample(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
        """Get common sample between two DataFrames.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame
        
        Returns:
            pd.Index: Common sample index
        """
        common_sample = df1.index.intersection(df2.index)
        df1 = df1.loc[common_sample]
        df2 = df2.loc[common_sample]
        return df1, df2


    @staticmethod
    def parse_date(date_str: str) -> pd.Timestamp:
        """Parse date string in YYYYmM format."""
        year = int(date_str[:4])
        month = int(date_str[5:])  # Skip the 'm'
        return pd.Timestamp(year=year, month=month, day=1)
