
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeClassifier
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# /////
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
def load_data():

    df = pd.read_csv(
        "data/processed/final_training_data.csv"
    )

    return df

 

def split_data(df):

    X = df.drop(
        columns=["is_high_risk"]
    )

    y = df["is_high_risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

  

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = (
        model.predict_proba(X_test)
        [:, 1]
    )

    metrics = {
        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),
        "precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
        "recall":
            recall_score(
                y_test,
                y_pred
            ),
        "f1":
            f1_score(
                y_test,
                y_pred
            ),
        "roc_auc":
            roc_auc_score(
                y_test,
                y_prob
            ),
    }

    return metrics

def train_and_log_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    

    with mlflow.start_run(
    run_name=model_name
):

     model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    mlflow.log_metrics(
        metrics
    )

    
    mlflow.log_param(
    "model_type",
    model_name
    )

    mlflow.sklearn.log_model(
            model,
            model_name
        )

    print(
        model_name,
        metrics
    )

 
def main():
    print("Inside main")

    df = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    models = {
        "LogisticRegression":
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),

        "DecisionTree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "RandomForest":
            RandomForestClassifier(
                random_state=42,
                n_estimators=100
            ),

        "GradientBoosting":
            GradientBoostingClassifier(
                random_state=42
            ),
    }

    for name, model in models.items():

        if mlflow.active_run():
            mlflow.end_run()
    

        train_and_log_model(
            model,
            name,
            X_train,
            X_test,
            y_train,
            y_test
        )
if __name__ == "__main__":
    print("Starting training...")

    main()
 

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(
    X_train,
    y_train
)

print(grid_search.best_params_)
print(grid_search.best_score_)