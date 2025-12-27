from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os
import pickle
import numpy as np
from uvicorn import run

templates = Jinja2Templates(directory="templates")
app = FastAPI()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

with open("../Models/breast-cancer-model.pkl", 'rb') as file:
    model_data = pickle.load(file)

w_final = model_data["w"]
b_final = model_data["b"]
mu = model_data["mu"]
sigma = model_data["sigma"]

def linear(X, w, b):
    return np.dot(X, w) + b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return (1 / (1 + np.exp(-z)))

@app.post("/predict")
def predict(
    request: Request,
    radius_mean: float = Form(...),
    texture_mean: float = Form(...),
    perimeter_mean: float = Form(...),
    area_mean: float = Form(...),
    smoothness_mean: float = Form(...),
    compactness_mean: float = Form(...),
    concavity_mean: float = Form(...),
    concave_points_mean: float = Form(...),
    symmetry_mean: float = Form(...),
    fractal_dimension_mean: float = Form(...),
    radius_se: float = Form(...),
    texture_se: float = Form(...),
    perimeter_se: float = Form(...),
    area_se: float = Form(...),
    smoothness_se: float = Form(...),
    compactness_se: float = Form(...),
    concavity_se: float = Form(...),
    concave_points_se: float = Form(...),
    symmetry_se: float = Form(...),
    fractal_dimension_se: float = Form(...)
):

    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se
    ]


    final_features = np.array(input_data).reshape(1, -1)
    final_features = (final_features - mu) / sigma
    print(f"Received Data:")

    z = linear(final_features, w_final, b_final)
    prediction = sigmoid(z)
    print(F"{prediction}")
    result = 1 if prediction >= 0.5 else 0
    
    if result == 1:
        text = "Prediction: Malignant (Cancerous)"
        color = "danger"  # Bootstrap class for Red
    else:
        text = "Prediction: Benign (Safe)"
        color = "success" # Bootstrap class for Green

    # 4. Return the HTML with the result variables
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction_text": text,
        "alert_type": color
    })
    



if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8000)