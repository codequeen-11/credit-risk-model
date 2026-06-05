from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd

app = FastAPI(
    title="Credit Risk API"
)

model = joblib.load(
    "../models/random_forest.pkl"
)