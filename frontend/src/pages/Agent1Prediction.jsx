import { useNavigate } from "react-router-dom";
import HeartView from "../components/HeartView";
import "./Agent.css";

export default function Agent1Prediction() {
  const navigate = useNavigate();

  // Safely read session storage
  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : null;

  if (!data || !data.agent1) {
    return (
      <div className="agent-container">
        <h2>No prediction data available</h2>

        <button
          className="agent-btn"
          onClick={() => navigate("/upload")}
        >
          ← Back to Upload
        </button>
      </div>
    );
  }

  const agentOutput = data.agent1;

  return (
  <div className="agent-container">

    {/* PARALLAX BACKGROUND */}
    <div className="parallax-bg"></div>

    {/* AI SCAN OVERLAY */}
    <div className="scan-overlay"></div>

    {/* HEADER */}
    <div className="agent-header">
      <h2>Agent 1 : Prediction Phase</h2>
      <span>AORTIX — Cardiovascular Digital Twin</span>
    </div>

    {/* DASHBOARD STATS */}
    <div className="dashboard-grid">
      <div className="stat-card">
        <h4>Prediction Score</h4>
        <p className="stat-value danger">
          {agentOutput?.risk_score || "78%"}
        </p>
      </div>

      <div className="stat-card">
        <h4>Status</h4>
        <p className="stat-value">
          {agentOutput?.status || "Active"}
        </p>
      </div>

      <div className="stat-card">
        <h4>Confidence</h4>
        <p className="stat-value">
          {agentOutput?.confidence || "92%"}
        </p>
      </div>
    </div>

    {/* ECG ANIMATED BACKGROUND */}
    <div className="ecg-bg"></div>

    {/* CONTENT */}
    <div className="agent-content">

      {/* HEART VIEW */}
      <div className="agent-left">
        <HeartView outputs={agentOutput} />
      </div>

      {/* ANALYSIS PANEL */}
      <div className="agent-right">
        <h3>Component-wise Analysis</h3>

        {agentOutput && Object.keys(agentOutput).length > 0 ? (
          Object.keys(agentOutput).map((k) => (
            <p key={k} style={{ marginBottom: "8px" }}>
              {agentOutput[k]}
            </p>
          ))
        ) : (
          <p>No prediction data received.</p>
        )}
      </div>
    </div>

    {/* CONTROLS */}
    <div className="agent-controls">

      <button
        className="agent-btn secondary"
        onClick={() => navigate("/upload")}
      >
        ← Back
      </button>

      <button
        className="agent-btn"
        onClick={() => navigate("/agent2")}
      >
        Next →
      </button>

    </div>

  </div>
);
}
