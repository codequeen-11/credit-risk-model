import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.feature_engineering import (
    AggregateFeatures,
    DateTimeFeatures,
    DropColumns,
)
from sklearn import set_config

set_config(transform_output="pandas")

categorical_features = [
    "CurrencyCode",
    "ProviderId",
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
]


numerical_features = [
    "Amount",
    "Value",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
]

DROP_COLUMNS = [
    "TransactionId",
    "BatchId",
    "AccountId",
    "SubscriptionId",
    "CustomerId",
    "ProductId",
    "TransactionStartTime"
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ],
     sparse_threshold=0
)

full_pipeline = Pipeline(
    steps=[
        (
            "aggregate_features",
            AggregateFeatures(),
        ),
        (
            "datetime_features",
            DateTimeFeatures(),
        ),
         (
            "drop_columns",
            DropColumns(DROP_COLUMNS),
        ),
        (
            "preprocessor",
            preprocessor,
        ),
    ]
)

 
def process_data(df):

    pipeline = build_pipeline()

    transformed = pipeline.fit_transform(df)

    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names
    )

    return transformed_df
def build_pipeline():
    """
    Return full preprocessing pipeline.
    """
    return full_pipeline