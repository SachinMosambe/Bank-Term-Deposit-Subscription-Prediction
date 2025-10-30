# feature_engineer.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Custom engineered features
        X['balance_per_age'] = X['balance'] / (X['age'] + 1)
        X['duration_per_campaign'] = X['duration'] / (X['campaign'] + 1)
        X['has_previous'] = (X['previous'] > 0).astype(int)

        return X
