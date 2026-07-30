import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


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

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved metrics.json")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
    )

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    print("Saved confusion_matrix.png")

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
    )

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()

    print("Saved roc_curve.png")

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))

    importances.head(15).sort_values().plot(kind="barh")

    plt.xlabel("Feature Importance")
    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()

    print("Saved feature_importance.png")

    print("\nDashboard assets generated successfully!")


if __name__ == "__main__":
    main()