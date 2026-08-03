from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictionRequest, PredictionResponse, ModelInfoResponse, ModelMetricsResponse, DriftReportResponse, RetrainResponse
from model_service import predict as predict_service, get_model_info, get_model_metrics, get_drift_report, trigger_retrain, reload_bundle

app = FastAPI(title="DriftOps API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server default port
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        return predict_service(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info_endpoint():
    return get_model_info()

@app.get("/model-metrics", response_model=ModelMetricsResponse)
def model_metrics_endpoint():
    return get_model_metrics()

@app.get("/drift-report", response_model=DriftReportResponse)
def drift_report_endpoint():
    return get_drift_report()

@app.post("/retrain", response_model=RetrainResponse)
def retrain_endpoint():
    result = trigger_retrain()
    if result["decision"] == "promoted":
        reload_bundle()  # force next predict() to pick up the newly promoted model
    return result