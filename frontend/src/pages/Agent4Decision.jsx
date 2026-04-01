import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent4Decision() {

  const navigate = useNavigate();

  /* ===============================
     SAFE DATA FETCH
  =============================== */
  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : {};

  /* ===============================
     DEFAULT DEMO DATA (IMPORTANT)
  =============================== */
  const agent4 = data?.agent4 || {
    final_treatment: "Lifestyle + Medication",
    decision: "Moderate Risk – Preventive Care Required",
    strategy: "Regular monitoring, diet improvement, and medication adherence",
    monitoring: "Weekly BP check, monthly ECG review",
    confidence: "91%",
    risk_level: "Moderate"
  };

  return (
    <div className="agent-container">

      {/* BACKGROUND EFFECTS */}
      <div className="parallax-bg"></div>
      <div className="scan-overlay"></div>

      {/* ===============================
         HEADER
      =============================== */}
      <div className="agent-header">
        <h2>Agent 4 : Final Decision</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* ===============================
         CONTENT
      =============================== */}
      <div className="agent-content">

        {/* LEFT — FINAL DECISION */}
        <div className="agent-left">

          <div className="stat-card">
            <h3>Final Treatment</h3>
            <p className="stat-value">
              {agent4.final_treatment}
            </p>
          </div>

          <div className="stat-card">
            <h3>Decision</h3>
            <p className="stat-value danger">
              {agent4.decision}
            </p>
          </div>

          <div className="stat-card">
            <h3>Strategy</h3>
            <p>{agent4.strategy}</p>
          </div>

          <div className="stat-card">
            <h3>Monitoring Plan</h3>
            <p>{agent4.monitoring}</p>
          </div>

        </div>

        {/* RIGHT — METRICS */}
        <div className="agent-right">

          <div className="stat-card">
            <h3>Confidence</h3>
            <p className="stat-value">
              {agent4.confidence}
            </p>
          </div>

          <div className="stat-card">
            <h3>Risk Level</h3>
            <p className="stat-value danger">
              {agent4.risk_level}
            </p>
          </div>

        </div>

      </div>

      {/* ===============================
         CONTROLS
      =============================== */}
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
          Generate Final Report →
        </button>

      </div>

    </div>
  );
}