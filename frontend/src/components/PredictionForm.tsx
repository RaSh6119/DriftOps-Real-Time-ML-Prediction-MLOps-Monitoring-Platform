import { useState } from "react";
import { getPrediction } from "../api/client";
import type { PredictionRequest, PredictionResponse } from "../types";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

interface Props {
  onResult: (result: PredictionResponse) => void;
}

export default function PredictionForm({ onResult }: Props) {
  const [form, setForm] = useState<PredictionRequest>({
    origin: "",
    destination: "",
    carrier: "",
    distance_miles: 0,
    scheduled_hour: 12,
    day_of_week: 1,
    month: 1,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof PredictionRequest>(key: K, value: PredictionRequest[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await getPrediction({
        ...form,
        origin: form.origin.toUpperCase(),
        destination: form.destination.toUpperCase(),
        carrier: form.carrier.toUpperCase(),
      });
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6 space-y-4">
      <h2 className="text-lg font-semibold text-gray-800">Trip details</h2>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Origin airport</label>
          <input
            required maxLength={3} placeholder="ATL"
            value={form.origin}
            onChange={(e) => update("origin", e.target.value)}
            className="w-full border rounded px-3 py-2 uppercase"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Destination airport</label>
          <input
            required maxLength={3} placeholder="ORD"
            value={form.destination}
            onChange={(e) => update("destination", e.target.value)}
            className="w-full border rounded px-3 py-2 uppercase"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Carrier</label>
          <input
            required maxLength={2} placeholder="DL"
            value={form.carrier}
            onChange={(e) => update("carrier", e.target.value)}
            className="w-full border rounded px-3 py-2 uppercase"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Distance (miles)</label>
          <input
            required type="number" min={1}
            value={form.distance_miles || ""}
            onChange={(e) => update("distance_miles", Number(e.target.value))}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Scheduled hour (0-23)</label>
          <input
            required type="number" min={0} max={23}
            value={form.scheduled_hour}
            onChange={(e) => update("scheduled_hour", Number(e.target.value))}
            className="w-full border rounded px-3 py-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Day of week</label>
          <select
            value={form.day_of_week}
            onChange={(e) => update("day_of_week", Number(e.target.value))}
            className="w-full border rounded px-3 py-2"
          >
            {DAYS.map((d, i) => (
              <option key={d} value={i + 1}>{d}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-600 mb-1">Month</label>
          <select
            value={form.month}
            onChange={(e) => update("month", Number(e.target.value))}
            className="w-full border rounded px-3 py-2"
          >
            {MONTHS.map((m, i) => (
              <option key={m} value={i + 1}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <p className="text-red-600 text-sm">{error}</p>}

      <button
        type="submit" disabled={loading}
        className="bg-blue-600 text-white px-4 py-2 rounded font-medium disabled:opacity-50"
      >
        {loading ? "Predicting..." : "Predict delay"}
      </button>
    </form>
  );
}