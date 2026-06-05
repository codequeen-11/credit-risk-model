from fastapi import FastAPI

import joblib
import pandas as pd

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)
 
from fastapi import FastAPI, HTTPException
import traceback

from pathlib import Path
app = FastAPI(
    title="Credit Risk API"
)

 

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "credit_risk_pipeline.pkl"

print("Loading model from:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

 


@app.get("/")
def home():

    return {
        "message":
        "Credit Risk Model API Running"
    }

 


@app.post("/predict")
def predict(request: PredictionRequest):

    try:
        input_df = pd.DataFrame(
            [request.dict()]
             
        )

        print("Input Shape:", input_df.shape)
        print("Columns:", input_df.columns.tolist())

        probability = model.predict_proba(
            input_df
        )[0][1]
#     

        prediction = (
            "High Risk"
            if probability >= 0.5
            else "Low Risk"
        )

        return {
            "risk_probability": float(probability),
            "prediction": prediction
        }

    except Exception as e:

        print("ERROR:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    