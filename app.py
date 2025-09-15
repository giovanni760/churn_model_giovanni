from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Churn Prediction')

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# 1. Cargar tu modelo entrenado de churn
model = load(pathlib.Path('model/churn-model-v1.joblib'))

# 2. Definir las features que espera el modelo
class InputData(BaseModel):
    CreditScore: int = 600
    Age: int = 40
    Tenure: int = 5
    Balance: float = 60000.0
    NumOfProducts: int = 2
    HasCrCard: int = 1
    IsActiveMember: int = 1
    EstimatedSalary: float = 50000.0
    Geography_Germany: int = 0
    Geography_Spain: int = 0
    Gender_Male: int = 1  # (0=femenino, 1=masculino)

# 3. Definir el output
class OutputData(BaseModel):
    score: float = 0.0

# 4. Endpoint de predicción
@app.post('/score', response_model=OutputData)
def score(data: InputData):
    # Convertir input a numpy array en el mismo orden que el entrenamiento
    model_input = np.array([v for v in data.dict().values()]).reshape(1, -1)

    # Probabilidad de que el cliente se dé de baja (Exited=1)
    result = model.predict_proba(model_input)[:, -1]

    return {'score': float(result[0])}

