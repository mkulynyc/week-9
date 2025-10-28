import pandas as pd


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
        # Merge X_ with the grouped data to get the estimates
        merged = pd.merge(X_, self.grouped_data, how='left', on=list(X_.columns))

        # Count how many rows have NaN in the target column
        missing_count = merged['target'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} observations have missing groups and will return NaN.")

        return merged['target'].values
    


### RUN testcode IN THE TERMINAL TO CHECK