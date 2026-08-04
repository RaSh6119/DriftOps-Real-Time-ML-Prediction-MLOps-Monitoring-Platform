import type {
  PredictionRequest, PredictionResponse, ModelInfoResponse,
  ModelMetricsResponse, DriftReportResponse, RetrainResponse,
} from "../types";

const BASE_URL = "http://localhost:8000";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export async function getPrediction(req: PredictionRequest): Promise<PredictionResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  return handleResponse<PredictionResponse>(res);
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return handleResponse(await fetch(`${BASE_URL}/model-info`));
}

export async function getModelMetrics(): Promise<ModelMetricsResponse> {
  return handleResponse(await fetch(`${BASE_URL}/model-metrics`));
}

export async function getDriftReport(): Promise<DriftReportResponse> {
  return handleResponse(await fetch(`${BASE_URL}/drift-report`));
}

export async function triggerRetrain(): Promise<RetrainResponse> {
  return handleResponse(await fetch(`${BASE_URL}/retrain`, { method: "POST" }));
}