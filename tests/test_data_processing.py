import pandas as pd
from sklearn.pipeline import Pipeline
from src.feature_engineering import (
    DateTimeFeatures,
    AggregateFeatures,
)

from src.data_processing import (
    build_pipeline,
    process_data,
    create_proxy_target,
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
    assert "transaction_month" in result.columns
    assert "transaction_year" in result.columns
 
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

    # assert (
    #     "total_transaction_amount"
    #     in result.columns
    # )
    assert "total_transaction_amount" in result.columns
    assert "avg_transaction_amount" in result.columns
    assert "transaction_count" in result.columns


def test_build_pipeline_returns_pipeline():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)


def test_process_data_returns_dataframe():
    sample_df = pd.DataFrame(
        {
            "TransactionId": [1],
            "BatchId": [1],
            "AccountId": [1],
            "SubscriptionId": [1],
            "CustomerId": [1],
            "ProductId": [1],
            "TransactionStartTime": ["2025-01-01 10:00:00"],
            "CurrencyCode": ["USD"],
            "ProviderId": ["Provider1"],
            "ProductCategory": ["Utility"],
            "ChannelId": ["Web"],
            "PricingStrategy": [1],
            "Amount": [100],
            "Value": [100],
        }
    )

    result = process_data(sample_df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_create_proxy_target():
    sample_df = pd.DataFrame(
        {
            "CustomerId": [1, 1, 2, 2 ,3., 3],
            "TransactionId": [1, 2, 3, 4, 5, 6],
            "TransactionStartTime": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-05",
                "2025-01-06",
                "2025-01-08",
                "2025-01-09",
            ],
            "Value": [100, 150, 50, 75, 300, 250],
        }
    )

    result = create_proxy_target(sample_df)

    assert "is_high_risk" in result.columns
    assert set(result["is_high_risk"].unique()).issubset({0, 1})
