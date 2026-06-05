from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import build_pipeline

def build_model_pipeline():

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_pipeline()
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    n_estimators=100
                )
            )
        ]
    )

    return model_pipeline