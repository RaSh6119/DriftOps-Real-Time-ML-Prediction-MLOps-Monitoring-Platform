import { useEffect, useState } from "react";
import { getDriftReport, triggerRetrain } from "../api/client";
import type { DriftReportResponse, RetrainResponse } from "../types";

function severityColor(severity: string): string {
  switch (severity) {
    case "high": return "bg-red-100 text-red-800";
    case "moderate": return "bg-yellow-100 text-yellow-800";
    default: return "bg-green-100 text-green-800";
  }
}

export default function DriftPanel() {
  const [report, setReport] = useState<DriftReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainResult, setRetrainResult] = useState<RetrainResponse | null>(null);

  function loadReport() {
    setError(null);
    getDriftReport()
      .then(setReport)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load drift report"));
  }

  useEffect(() => {
    loadReport();
  }, []);

  async function handleRetrain() {
    setRetraining(true);
    setRetrainResult(null);
    try {
      const result = await triggerRetrain();
      setRetrainResult(result);
      loadReport();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Retrain failed");
    } finally {
      setRetraining(false);
    }
  }

  if (error) return <p className="text-red-600 text-sm">{error}</p>;
  if (!report) return <p className="text-gray-500 text-sm">Loading drift report...</p>;

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Data drift</h2>
        <span className={`text-xs font-medium px-2 py-1 rounded ${severityColor(report.severity)}`}>
          {report.severity}
        </span>
      </div>

      <p className="text-sm text-gray-600">
        {report.drift_detected
          ? `Drift detected in: ${report.drifted_features.join(", ")}`
          : "No significant drift detected."}
      </p>

      <div className="space-y-1">
        {Object.entries(report.per_feature).map(([feature, data]) => (
          <div key={feature} className="flex justify-between text-sm border-b py-1">
            <span className="text-gray-700">{feature}</span>
            <span className="flex items-center gap-2">
              <span className={`text-xs px-1.5 py-0.5 rounded ${severityColor(data.severity)}`}>
                PSI {data.psi}
              </span>
              {data.reference_mean !== null && (
                <span className="text-gray-400 text-xs">
                  {data.reference_mean} → {data.current_mean}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>

      <p className="text-sm text-gray-700">
        <span className="font-medium">Recommendation:</span> {report.recommendation}
      </p>

      <button
        onClick={handleRetrain}
        disabled={retraining}
        className="bg-gray-800 text-white px-4 py-2 rounded font-medium text-sm disabled:opacity-50"
      >
        {retraining ? "Retraining..." : "Trigger retrain"}
      </button>

      {retrainResult && (
        <div className="text-sm bg-gray-50 rounded p-3">
          <p>Decision: <span className="font-medium">{retrainResult.decision}</span></p>
          <p className="text-gray-500 text-xs">
            Candidate v{retrainResult.candidate_version} (ROC-AUC {retrainResult.candidate_metrics.roc_auc?.toFixed(4)})
            {" vs. "}production v{retrainResult.current_production_version ?? "—"}
            {retrainResult.current_production_roc_auc != null &&
              ` (ROC-AUC ${retrainResult.current_production_roc_auc.toFixed(4)})`}
          </p>
        </div>
      )}
    </div>
  );
}