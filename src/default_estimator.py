import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
def create_proxy_target(df, n_clusters=3):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"]
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=(
                "TransactionId",
                "count"
            ),
            Monetary=(
                "Value",
                "sum"
            )
        )
        .reset_index()
    )

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    rfm["cluster"] = kmeans.fit_predict(
        rfm_scaled
    )

    cluster_summary = (
        rfm.groupby("cluster")
        [["Recency", "Frequency", "Monetary"]]
        .mean()
    )

    high_risk_cluster = (
        cluster_summary
        .sort_values(
            by=["Recency", "Frequency", "Monetary"],
            ascending=[False, True, True]
        )
        .index[0]
    )

    rfm["is_high_risk"] = (
        rfm["cluster"] == high_risk_cluster
    ).astype(int)

    return rfm

def create_training_data(df, processed_df):

    target_df = create_proxy_target(df)[
        ["CustomerId", "is_high_risk"]
    ]

    processed_df["CustomerId"] = df["CustomerId"].values

    final_df = processed_df.merge(
        target_df,
        on="CustomerId",
        how="left"
    )

    return final_df

def create_transaction_level_target(df):

    rfm = create_proxy_target(df)

    target_df = rfm[
        ["CustomerId", "is_high_risk"]
    ]

    final_df = df.merge(
        target_df,
        on="CustomerId",
        how="left"
    )

    return final_df