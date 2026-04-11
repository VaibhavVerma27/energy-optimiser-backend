"""API Router: /api/status"""
from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter()

@router.get("/")
def get_status():
    return {
        "api": "online",
        "model_trained": os.path.exists("model.joblib"),
        "timestamp": datetime.utcnow().isoformat(),
    }