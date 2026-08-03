from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    origin: str = Field(..., description="Origin airport IATA code, e.g. ATL")
    destination: str = Field(..., description="Destination airport IATA code, e.g. ORD")
    carrier: str = Field(..., description="Airline code, e.g. DL")
    distance_miles: float
    scheduled_hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=1, le=7)
    month: int = Field(..., ge=1, le=12)

class FeatureContribution(BaseModel):
    feature: str
    label: str
    impact_pct: float
    direction: str

class PredictionResponse(BaseModel):
    delay_probability: float
    predicted_delay_minutes: float
    confidence: str
    top_features: List[FeatureContribution]
    model_version: str | None
    latency_ms: float