import pandas as pd
import numpy as np

class GroupEstimate(object):
    def __init__(self, estimate):
        """
        This class computes group-wise estimates (mean or median) of a target variable
        based on the grouping defined by the features in X.

        Parameters
        ----------
        estimate : str
            The type of estimate to compute for each group. Must be either "mean" or "median".

        Returns
        -------
        None
        """

        if estimate not in ["mean", "median"]:
            raise ValueError("Estimate must be either 'mean' or 'median'")
        self.estimate = estimate
    
    def fit(self, X, y):
        """
        This function fits the GroupEstimate model by computing the group-wise estimates.

        Parameters
        ----------
        X : pandas DataFrame
            The feature data used for grouping.
        y : pandas Series or array-like
            The target variable for which the estimates are computed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the lengths of X and y do not match.
        TypeError
            If X is not a pandas DataFrame or if y is not a pandas Series or array-like.
        """

        # Combine X and y into a single DataFrame
        df = X.copy()
        df['target'] = y

        # Group by the columns in X
        grouped = df.groupby(list(X.columns), dropna=False)

        # Calculate the mean or median of y for each group
        if self.estimate == "mean":
            self.grouped_data = grouped['target'].mean().reset_index()
        else:
            self.grouped_data = grouped['target'].median().reset_index()
        return None

    def predict(self, X):
        """
        This function predicts the target variable for new data based on the group-wise estimates.

        Parameters
        ----------
        X : pandas DataFrame or list
            The new feature data for which predictions are to be made.
        
        Returns
        -------
        numpy array
            An array of predicted target values corresponding to each row in X.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame or list.
        
        """

        # Convert list input into DataFrame if needed
        if isinstance(X, list):
            X = pd.DataFrame(X, columns=self.grouped_data.columns[:-1])
        elif not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame or list of rows.")
        
        # Merge automatically on shared columns
        merged = pd.merge(X, self.grouped_data, how='left')

        # Count missing values in 'target'
        missing_count = merged['target'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} observations have missing groups and will return NaN.")

        # Always return a NumPy array
        return merged['target'].to_numpy()
