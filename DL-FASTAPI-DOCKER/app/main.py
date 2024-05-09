from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from fastapi.exceptions import RequestValidationError
from app.model.model import __version__ as model_version


app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
	
    predicted_label = predict_pipeline(payload.text)
    return PredictionOut(label = predicted_label)


