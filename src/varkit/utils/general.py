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
