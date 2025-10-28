import pandas as pd
import numpy as np

class GroupEstimate(object):
    def __init__(self, estimate):
        if estimate not in ["mean", "median"]:
            raise ValueError("Estimate must be either 'mean' or 'median'")
        self.estimate = estimate
    
    def fit(self, X, y):
        # Combine X and y into a single DataFrame
        df = X.copy()
        df['target'] = y

        # Group by the columns in X
        grouped = df.groupby(list(X.columns))

        # Calculate the mean or median of y for each group
        if self.estimate == "mean":
            self.grouped_data = grouped['target'].mean().reset_index()
        else:  # median
            self.grouped_data = grouped['target'].median().reset_index()
        return None

    def predict(self, X):
        # Handle both DataFrame and list input
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
            merged = pd.merge(X, self.grouped_data, how='left', on=cols)
        elif isinstance(X, list):
            # If X is a list of column names, extract those columns from grouped_data
            merged = self.grouped_data[self.grouped_data.columns.intersection(X + ['target'])]
        else:
            raise TypeError("X must be a pandas DataFrame or list of column names.")

        # Count how many rows have NaN in the target column (only if target exists)
        if 'target' in merged.columns:
            missing_count = merged['target'].isna().sum()
            if missing_count > 0:
                print(f"Warning: {missing_count} observations have missing groups and will return NaN.")
            return merged['target'].values
        else:
            # if target not found, return NaN array
            return np.full(len(X), np.nan)
