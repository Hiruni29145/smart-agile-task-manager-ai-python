"""
main.py

FastAPI application for the
Smart Agile Task Manager AI.
"""

from fastapi import FastAPI

from schemas import PredictionRequest

from predict import predict


app = FastAPI(

    title="Smart Agile Task Manager AI",

    version="1.0.0",

    description="Story Point & Hours Prediction API",
)


@app.get("/")
def root():

    return {

        "status": "running",

        "message": "Smart Agile Task Manager AI API"

    }


@app.post("/api/predict")
def estimate_task(request: PredictionRequest):

    result = predict(

        task_name=request.task_name,

        description=request.description,

        priority=request.priority,

        task_type=request.task_type,

    )

    return result