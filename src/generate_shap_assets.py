from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)
SHAP_SUMMARY = Path("reports/shap_summary.png")

def load_data():
    return pd.read_csv("data/processed/final_training_data.csv")


def main():

    print("Loading data...")

    df = load_data()

    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Loading model...")

    model = joblib.load("models/random_forest.pkl")

    print("Generating SHAP values...")

    # Use a sample to keep generation fast
    sample = X_test.sample(
        min(500, len(X_test)),
        random_state=42,
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()

    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values[1],
            sample,
            show=False,
        )
    else:
        shap.summary_plot(
            shap_values,
            sample,
            show=False,
        )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved reports/shap_summary.png")


if __name__ == "__main__":
    main()