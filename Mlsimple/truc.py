from pydantic import BaseModel
import numpy as np
import joblib
from fastapi import FastAPI


# Charger le modèle
model = joblib.load("model.pkl")

app = FastAPI(title="API prédiction Y")

# Schéma des données entrantes
class InputData(BaseModel):
    A: float
    B: float

@app.post("/predict")

def predict(data: InputData):
    X = [[data.A, data.B]]
    prediction = model.predict(X)

    return {"prediction": float(prediction[0])}