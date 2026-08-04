import { useState } from "react";
import PredictionForm from "./components/PredictionForm";
import PredictionResult from "./components/PredictionResult";
import MonitoringDashboard from "./components/MonitoringDashboard";
import DriftPanel from "./components/DriftPanel";
import type { PredictionResponse } from "./types";

type Tab = "predict" | "monitoring";

function App() {
  const [tab, setTab] = useState<Tab>("predict");
  const [result, setResult] = useState<PredictionResponse | null>(null);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 mb-6">DriftOps</h1>

        <div className="flex gap-2 mb-6 border-b">
          <TabButton active={tab === "predict"} onClick={() => setTab("predict")}>
            Predict
          </TabButton>
          <TabButton active={tab === "monitoring"} onClick={() => setTab("monitoring")}>
            Monitoring
          </TabButton>
        </div>

        {tab === "predict" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <PredictionForm onResult={setResult} />
            {result && <PredictionResult result={result} />}
          </div>
        )}

        {tab === "monitoring" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <MonitoringDashboard />
            <DriftPanel />
          </div>
        )}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
        active ? "border-blue-600 text-blue-600" : "border-transparent text-gray-500 hover:text-gray-700"
      }`}
    >
      {children}
    </button>
  );
}

export default App;