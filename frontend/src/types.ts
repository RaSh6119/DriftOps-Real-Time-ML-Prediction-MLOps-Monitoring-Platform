export interface PredictionRequest {
  origin: string;
  destination: string;
  carrier: string;
  distance_miles: number;
  scheduled_hour: number;
  day_of_week: number;
  month: number;
}

export interface FeatureContribution {
  feature: string;
  label: string;
  impact_pct: number;
  direction: "increases" | "decreases";
}

export interface PredictionResponse {
  delay_probability: number;
  predicted_delay_minutes: number;
  confidence: "Low" | "Medium" | "High";
  top_features: FeatureContribution[];
  model_version: string | null;
  latency_ms: number;
}

export interface ModelInfoResponse {
  version: string | null;
  trained_at: string | null;
  metrics: Record<string, number>;
}

export interface VersionMetrics {
  version: string;
  trained_at: string;
  accuracy: number | null;
  roc_auc: number | null;
  f1: number | null;
}

export interface ModelMetricsResponse {
  live_prediction_count: number;
  failed_predictions: number;
  avg_latency_ms: number;
  current_model_version: string | null;
  accuracy_history: VersionMetrics[];
}

export interface FeatureDrift {
  psi: number;
  severity: "none" | "moderate" | "high";
  reference_mean: number | null;
  current_mean: number | null;
}

export interface DriftReportResponse {
  drift_detected: boolean;
  severity: "none" | "moderate" | "high";
  overall_psi: number;
  drifted_features: string[];
  per_feature: Record<string, FeatureDrift>;
  recommendation: string;
}

export interface RetrainResponse {
  decision: "promoted" | "kept_existing";
  candidate_version: string;
  candidate_metrics: Record<string, number>;
  current_production_version: string | null;
  current_production_roc_auc: number | null;
}