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

class ModelInfoResponse(BaseModel):
    version: str | None
    trained_at: str | None
    metrics: dict

class VersionMetrics(BaseModel):
    version: str
    trained_at: str
    accuracy: float | None
    roc_auc: float | None
    f1: float | None

class ModelMetricsResponse(BaseModel):
    live_prediction_count: int
    failed_predictions: int
    avg_latency_ms: float
    current_model_version: str | None
    accuracy_history: List[VersionMetrics]

class FeatureDrift(BaseModel):
    psi: float
    severity: str
    reference_mean: float | None
    current_mean: float | None

class DriftReportResponse(BaseModel):
    drift_detected: bool
    severity: str
    overall_psi: float
    drifted_features: List[str]
    per_feature: dict[str, FeatureDrift]
    recommendation: str

class RetrainResponse(BaseModel):
    decision: str
    candidate_version: str
    candidate_metrics: dict
    current_production_version: str | None
    current_production_roc_auc: float | None