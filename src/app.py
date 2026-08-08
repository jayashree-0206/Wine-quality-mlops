from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained best model
model = joblib.load("models/best_model.pkl")

# Create FastAPI application
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Predicts wine quality using a trained Random Forest model",
    version="1.0"
)


# Input data format
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Wine Quality Prediction API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: WineData):

    input_data = pd.DataFrame([{
        "fixed acidity": data.fixed_acidity,
        "volatile acidity": data.volatile_acidity,
        "citric acid": data.citric_acid,
        "residual sugar": data.residual_sugar,
        "chlorides": data.chlorides,
        "free sulfur dioxide": data.free_sulfur_dioxide,
        "total sulfur dioxide": data.total_sulfur_dioxide,
        "density": data.density,
        "pH": data.ph,
        "sulphates": data.sulphates,
        "alcohol": data.alcohol
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_quality": round(float(prediction), 2)
    }