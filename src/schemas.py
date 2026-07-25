"""
schemas.py

Pydantic models for FastAPI.
"""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    task_name: str
    description: str
    priority: str
    task_type: str