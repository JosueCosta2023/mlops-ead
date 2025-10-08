from fastapi import FastAPI
from pydantic import BaseModel

import os
import json
import mlflow
import uvicorn
import numpy as np

app = FastAPI(
    title="Unidade 4 - Fetal Helath",
    description="Aulas Praticas",
    version="1.0",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Routes get Api"
        },
        {
            "name":"Prediction",
            "description":"Routes Predictions possibilities"
        }
    ])

def load_model():
    print("Lendo modelo...")
    MLFLOW_TRACKING_URI='https://dagshub.com/JosueCosta2023/mlops-ead.mlflow'
    MLFLOW_TRACKING_USERNAME="JosueCosta2023"
    MLFLOW_TRACKING_PASSWORD="6872c427bf234a7f2f5fab8e325bcac9efcc77c1"
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

    print("Configurando mlflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("Criando cliente...")
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    print("Buscando o registro do modelo...")
    registered_model= client.get_registered_model("fetal_health")

    print("Lendo modelo")
    run_id= registered_model.latest_versions[-1].run_id

    logged_model= f'runs:/{run_id}/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    print(loaded_model)
    return loaded_model

@app.get(
    path="/",
    tags=["Health"]
)

def api_health():
    return {"status":"healthy"}

@app.post(
    path="/predict",
    tags=["Prediction"]
)
def predict():
    load_model()
    return {"status": "ok"}