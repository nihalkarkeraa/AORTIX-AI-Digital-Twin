import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent3Risk() {

  const navigate = useNavigate();

  /* ===============================
     SAFE DATA FETCH
  =============================== */
  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : {};

  /* ===============================
     DEFAULT DEMO DATA (IMPORTANT)
  =============================== */
  const agent3 = data?.agent3 || {
    status: "Stable",
    risk_level: "Moderate",
    risk_index: "0.62",
    confidence: "89%",
    best_treatment: "Medication + Lifestyle",
    treatments: [
      {
        treatment: "Medication",
        score: "92%",
        recommended: true,
        reason: "Effective for stabilizing cardiovascular conditions"
      },
      {
        treatment: "Lifestyle Change",
        score: "85%",
        recommended: false,
        reason: "Requires long-term discipline but beneficial"
      }
    ]
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
        <h2>Agent 3 : Risk & Safety</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* ===============================
         CONTENT
      =============================== */}
      <div className="agent-content">

        {/* LEFT — OVERALL RISK */}
        <div className="agent-left">

          <div className="stat-card">
            <h3>Status</h3>
            <p className="stat-value">{agent3.status}</p>
          </div>

          <div className="stat-card">
            <h3>Risk Level</h3>
            <p className="stat-value danger">{agent3.risk_level}</p>
          </div>

          <div className="stat-card">
            <h3>Risk Index</h3>
            <p className="stat-value">{agent3.risk_index}</p>
          </div>

          <div className="stat-card">
            <h3>Confidence</h3>
            <p className="stat-value">{agent3.confidence}</p>
          </div>

          <div className="stat-card">
            <h3>Best Treatment</h3>
            <p>{agent3.best_treatment}</p>
          </div>

        </div>

        {/* RIGHT — TREATMENT DETAILS */}
        <div className="agent-right">

          <h3 style={{ marginBottom: "15px" }}>
            Treatment Analysis
          </h3>

          {agent3.treatments && agent3.treatments.length > 0 ? (
            agent3.treatments.map((t, idx) => (
              <div className="stat-card" key={idx}>

                <p className="stat-value">
                  {t.treatment} → {t.score}
                  {t.recommended && " ✅"}
                </p>

                <p style={{ opacity: 0.7 }}>
                  {t.reason}
                </p>

              </div>
            ))
          ) : (
            <p>No treatment data available</p>
          )}

        </div>

      </div>

      {/* ===============================
         CONTROLS
      =============================== */}
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