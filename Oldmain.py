import pandas as pd
from fastapi import FastAPI


app = FastAPI(title="Fetal Health API", openapi_tags=[
    {
        "name" : "Health",
        "description" :"Get Api health"
    },
    {
        "name" : "Prediction",
        "description" :"Model Prediction "
    }
])
from fastapi import FastAPI
from tensorflow.python.platform import self_check


@app.get(path='/', tags=['Health'])

def api_health():
    return {"status":"healthy"}



@app.post(path='/predict', tags=['Prediction'])
def predict():
    return {"status":0}
