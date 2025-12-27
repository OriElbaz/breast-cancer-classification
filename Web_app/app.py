from fastapi import FastAPI
from starlette.responses import FileResponse
import os
import pickle
import numpy as np

app = FastAPI()

model_path = os.path.join('models', 'breast-cancer-model.pkl')
with open(model_path, 'rb') as file:
    model_data = pickle.load(file)

w_final = model_data["w"]
w_final = model_data["b"]
w_final = model_data["mu"]
w_final = model_data["sigma"]

def predict(X, w, b):
    return np.dot(X, w) + b

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return (1 / (1 + np.exp(-z)))

@app.get("/")
async def root():
    return FileResponse('./Templates/index.html')
