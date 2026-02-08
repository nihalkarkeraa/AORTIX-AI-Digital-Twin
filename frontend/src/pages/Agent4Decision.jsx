import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent4Decision() {

  const navigate = useNavigate();

  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : null;

  if (!data || !data.agent4) {
    return (
      <div className="agent-container">
        <h2>No Agent-4 decision available</h2>
        <button className="agent-btn" onClick={() => navigate("/agent3")}>
          ← Back
        </button>
      </div>
    );
  }

  const agent4 = data.agent4;

  return (
    <div className="agent-container">

      {/* HEADER */}
      <div className="agent-header">
        <h2>Agent 4 : Final Decision</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* CONTENT */}
      <div className="agent-content">

        {/* LEFT — FINAL DECISION */}
        <div className="agent-left">
          <h3>Final Strategy</h3>

          <p><strong>Treatment:</strong> {agent4.final_treatment}</p>
          <p><strong>Decision:</strong> {agent4.decision}</p>
          <p><strong>Strategy:</strong> {agent4.strategy}</p>
          <p><strong>Monitoring:</strong> {agent4.monitoring}</p>
        </div>

        {/* RIGHT — METRICS */}
        <div className="agent-right">
          <h3>Decision Metrics</h3>

          <p><strong>Confidence:</strong> {agent4.confidence}</p>
          <p><strong>Risk Level:</strong> {agent4.risk_level}</p>
        </div>

      </div>

      {/* CONTROLS */}
      <div className="agent-controls">

        <button
          className="agent-btn secondary"
          onClick={() => navigate("/agent3")}
        >
          ← Back
        </button>

        <button
          className="agent-btn"
          onClick={() => navigate("/final")}
        >
          Finish →
        </button>

      </div>

    </div>
  );
}
