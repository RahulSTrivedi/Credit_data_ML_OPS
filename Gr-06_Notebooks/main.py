#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="Credit ML Model API",
    description="API for Classify whether a customer credit rating is good or bad using the best ML model from MLflow",
    version="1.0.0"
)

# Define the expected input schema using Pydantic
class InputData(BaseModel):
    checking_account_status: Literal["A11", "A12", "A13", "A14"]
    duration: int = Field(..., gt=0)  # Must be greater than 0
    credit_history: Literal["A30", "A31", "A32", "A33", "A34"]
    purpose: Literal["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"]
    credit_amount: int = Field(..., gt=0)  # Must be greater than 0
    savings_account: Literal["A61", "A62", "A63", "A64", "A65"]
    employment_status: Literal["A71", "A72", "A73", "A74", "A75"]
    installment_rate: int = Field(..., ge=1, le=4)  # Must be between 1 and 4
    personal_status_sex: Literal["A91", "A92", "A93", "A94"]
    other_debtors: Literal["A101", "A102", "A103"]
    present_residence: int = Field(..., ge=1, le=4)  # Must be between 1 and 4
    property: Literal["A121", "A122", "A123", "A124"]
    age: int = Field(..., gt=0)  # Must be greater than 0
    other_installment_plans: Literal["A141", "A142", "A143"]
    housing: Literal["A151", "A152", "A153"]
    number_of_existing_credits: int = Field(..., ge=1, le=4)  # Must be between 1 and 4
    job: Literal["A171", "A172", "A173", "A174"]
    dependents: int = Field(..., ge=1, le=2)  # Must be between 1 and 2
    telephone: Literal["A191", "A192"]
    foreign_worker: Literal["A201", "A202"]

# Load the best model directly from MLflow artifact location
model_path = "D:/Rahul/jio-files/AIDS/Quarter 4/ML-ops/ML_ops_project/Gr-06_MLOPS_Project/Project Code/Gr-06_Notebooks/mlartifacts/542127534477010729/dc33c8a249c14ef49b487928f0a5c630/artifacts/model"
try:
    model = mlflow.sklearn.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

@app.post("/predict", summary="Get Prediction", response_description="Prediction result")
def predict(input_data: InputData):
    """
    Accepts a JSON object containing input features,
    converts it to a DataFrame, and returns the model prediction.
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {"prediction": prediction[0]}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))



