# Step1: FastAPI app to serve MPG prediction model
from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

#initialize FastAPI app
app = FastAPI(title="Vehicle MPG Prediction API")

#Load Model and feature names
model = joblib.load("mpg_random_forest_model.pkl")
feature_names = joblib.load("mpg_model_features.pkl")


# Define input schema
class CarData(BaseModel):
    horsepower: float
    weight: float
    acceleration: float
    displacement: float
    cylinders: int
    model_year: int


@app.get("/")
def home():
    return {"message": "Welcome to the Vehicle MPG Prediction API!"}

@app.post("/predict")
def predict_mpg(data: CarData):
    """
    Predict MPG for a vehicle given its features.
    Expects a JSON with keys: horsepower, weight, acceleration, displacement, cylinders, model_year
    """
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])[feature_names]
    prediction = model.predict(df)[0]
    return {"predicted_mpg": round(float(prediction), 2)}

