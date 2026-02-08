import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent3Risk() {

  const navigate = useNavigate();

  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : null;

  if (!data || !data.agent3) {
    return (
      <div className="agent-container">
        <h2>No Agent-3 data available</h2>
        <button className="agent-btn" onClick={() => navigate("/agent2")}>
          ← Back
        </button>
      </div>
    );
  }

  const agent3 = data.agent3;

  return (
    <div className="agent-container">

      {/* HEADER */}
      <div className="agent-header">
        <h2>Agent 3 : Risk & Safety</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* CONTENT */}
      <div className="agent-content">

        {/* LEFT — OVERALL RISK */}
        <div className="agent-left">
          <h3>Overall Assessment</h3>
          <p><strong>Status:</strong> {agent3.status}</p>
          <p><strong>Risk Level:</strong> {agent3.risk_level}</p>
          <p><strong>Risk Index:</strong> {agent3.risk_index}</p>
          <p><strong>Confidence:</strong> {agent3.confidence}</p>
          <p><strong>Best Treatment:</strong> {agent3.best_treatment}</p>
        </div>

        {/* RIGHT — TREATMENT DETAILS */}
        <div className="agent-right">
          <h3>Treatment Analysis</h3>

          {agent3.treatments.map((t, idx) => (
            <div key={idx} style={{ marginBottom: "12px" }}>
              <p>
                <strong>{t.treatment}</strong> → {t.score}
                {t.recommended && " ✅"}
              </p>
              <small>{t.reason}</small>
            </div>
          ))}
        </div>

      </div>

      {/* CONTROLS */}
      <div className="agent-controls">
        <button
          className="agent-btn secondary"
          onClick={() => navigate("/agent2")}
        >
          ← Back
        </button>

        <button
          className="agent-btn"
          onClick={() => navigate("/agent4")}
        >
          Next →
        </button>
      </div>

    </div>
  );
}
