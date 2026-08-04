import type { PredictionResponse } from "../types";

interface Props {
  result: PredictionResponse;
}

function probabilityColor(p: number): string {
  if (p >= 0.6) return "text-red-600";
  if (p >= 0.3) return "text-yellow-600";
  return "text-green-600";
}

function confidenceBadge(confidence: string): string {
  switch (confidence) {
    case "High": return "bg-blue-100 text-blue-800";
    case "Medium": return "bg-gray-100 text-gray-700";
    default: return "bg-gray-50 text-gray-500";
  }
}

export default function PredictionResult({ result }: Props) {
  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500">Predicted delay probability</p>
          <p className={`text-4xl font-bold ${probabilityColor(result.delay_probability)}`}>
            {(result.delay_probability * 100).toFixed(0)}%
          </p>
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded ${confidenceBadge(result.confidence)}`}>
          {result.confidence} confidence
        </span>
      </div>

      <div>
        <p className="text-sm text-gray-500">Estimated delay</p>
        <p className="text-xl font-semibold text-gray-800">{result.predicted_delay_minutes} minutes</p>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Why this prediction?</h3>
        <div className="space-y-2">
          {result.top_features.map((f) => (
            <div key={f.feature} className="flex items-center gap-3">
              <span className={f.direction === "increases" ? "text-red-500" : "text-green-500"}>
                {f.direction === "increases" ? "▲" : "▼"}
              </span>
              <div className="flex-1">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-700">{f.label}</span>
                  <span className="text-gray-500">{f.impact_pct}%</span>
                </div>
                <div className="w-full bg-gray-100 rounded h-1.5 mt-1">
                  <div
                    className={`h-1.5 rounded ${f.direction === "increases" ? "bg-red-400" : "bg-green-400"}`}
                    style={{ width: `${Math.min(f.impact_pct, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <p className="text-xs text-gray-400 border-t pt-3">
        Model version {result.model_version ?? "unknown"} · {result.latency_ms}ms
      </p>
    </div>
  );
}