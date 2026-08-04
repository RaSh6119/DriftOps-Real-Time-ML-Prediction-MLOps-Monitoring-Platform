import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getModelMetrics } from "../api/client";
import type { ModelMetricsResponse } from "../types";

export default function MonitoringDashboard() {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelMetrics()
      .then(setMetrics)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load metrics"));
  }, []);

  if (error) return <p className="text-red-600 text-sm">{error}</p>;
  if (!metrics) return <p className="text-gray-500 text-sm">Loading metrics...</p>;

  const chartData = metrics.accuracy_history.map((v) => ({
    version: `v${v.version}`,
    roc_auc: v.roc_auc,
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <h2 className="text-lg font-semibold text-gray-800">Model monitoring</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="Live predictions" value={metrics.live_prediction_count} />
        <Stat label="Failed predictions" value={metrics.failed_predictions} />
        <Stat label="Avg latency" value={`${metrics.avg_latency_ms.toFixed(1)}ms`} />
        <Stat label="Current version" value={metrics.current_model_version ?? "—"} />
      </div>

      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-2">ROC-AUC across model versions</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis dataKey="version" fontSize={12} />
            <YAxis domain={[0.6, 0.75]} fontSize={12} />
            <Tooltip />
            <Line type="monotone" dataKey="roc_auc" stroke="#2563eb" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-50 rounded p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold text-gray-800">{value}</p>
    </div>
  );
}