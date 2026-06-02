from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Generate customer-level aggregate features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert "CustomerId" in X.columns
        assert "Amount" in X.columns

        df = X.copy()

        customer_features = (
            df.groupby("CustomerId")
            .agg(
                total_transaction_amount=("Amount", "sum"),
                avg_transaction_amount=("Amount", "mean"),
                transaction_count=("TransactionId", "count"),
                std_transaction_amount=("Amount", "std"),
            )
            .reset_index()
        )

        df = df.merge(
            customer_features,
            on="CustomerId",
            how="left"
        )

        return df


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract datetime-based features.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        assert "TransactionStartTime" in X.columns

        df = X.copy()

        df["TransactionStartTime"] = pd.to_datetime(
            df["TransactionStartTime"]
        )

        df["transaction_hour"] = (
            df["TransactionStartTime"].dt.hour
        )

        df["transaction_day"] = (
            df["TransactionStartTime"].dt.day
        )

        df["transaction_month"] = (
            df["TransactionStartTime"].dt.month
        )

        df["transaction_year"] = (
            df["TransactionStartTime"].dt.year
        )

        return df
    
class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(
            columns=self.columns,
            errors="ignore"
        )
     