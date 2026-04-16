"""
Smart Grid Demand Forecasting & Response System
FastAPI Backend — Entry Point

Setup (do this once in PyCharm):
    Right-click the app/ folder → Mark Directory as → Sources Root

Run from INSIDE the app/ folder:
    uvicorn main:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import forecast, status, demand, capacity

app = FastAPI(
    title="Smart Grid AI API",
    description="AI-based electricity demand forecasting and demand-response system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(status.router, prefix="/api/status", tags=["Status"])
app.include_router(demand.router, prefix="/api/demand", tags=["Demand"])
app.include_router(capacity.router, prefix="/api/capacity", tags=["Capacity"])


@app.get("/")
def root():
    return {"message": "Smart Grid AI API is running", "docs": "/docs"}