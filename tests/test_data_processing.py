import pandas as pd

from src.feature_engineering import (
    DateTimeFeatures
)

def test_datetime_features():

    sample_df = pd.DataFrame(
        {
            "TransactionStartTime": [
                "2025-01-01 10:00:00"
            ]
        }
    )

    transformer = DateTimeFeatures()

    result = transformer.transform(
        sample_df
    )

    assert "transaction_hour" in result.columns
    assert "transaction_day" in result.columns
from src.feature_engineering import (
    AggregateFeatures
)
def test_aggregate_features():

    sample_df = pd.DataFrame(
        {
            "CustomerId": [1, 1],
            "TransactionId": [1, 2],
            "Amount": [100, 200]
        }
    )

    transformer = AggregateFeatures()

    result = transformer.transform(
        sample_df
    )

    assert (
        "total_transaction_amount"
        in result.columns
    )