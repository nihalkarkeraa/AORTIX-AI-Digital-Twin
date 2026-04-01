import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent2Treatment() {

  const navigate = useNavigate();

  /* ===============================
     SAFE DATA FETCH
  =============================== */
  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : {};

  /* ===============================
     DEFAULT DEMO DATA (IMPORTANT)
  =============================== */
  const agent2 = data?.agent2 || {
    best: {
      treatment: "ACE Inhibitors",
      risk_reduction: "35%"
    },
    worst: {
      treatment: "No Treatment",
      risk_reduction: "5%"
    }
  };

  const best = agent2.best;
  const worst = agent2.worst;

  return (
    <div className="agent-container">

      {/* BACKGROUND EFFECTS */}
      <div className="parallax-bg"></div>
      <div className="scan-overlay"></div>

      {/* ===============================
         HEADER
      =============================== */}
      <div className="agent-header">
        <h2>Agent 2 : Treatment Simulation</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* ===============================
         CONTENT
      =============================== */}
      <div className="agent-content">

        {/* LEFT — WORST */}
        <div className="agent-left">

          <div className="stat-card">
            <h3>Worst Option</h3>

            <p className="stat-value danger">
              {worst.treatment}
            </p>

            <p>Risk Reduction: {worst.risk_reduction}</p>
          </div>

        </div>

        {/* RIGHT — BEST */}
        <div className="agent-right">

          <div className="stat-card">
            <h3>Best Treatment</h3>

            <p className="stat-value">
              {best.treatment}
            </p>

            <p>Risk Reduction: {best.risk_reduction}</p>
          </div>

        </div>

      </div>

      {/* ===============================
         CONTROLS
      =============================== */}
      <div className="agent-controls">

        <button
          className="agent-btn secondary"
          onClick={() => navigate("/agent1")}
        >
          ← Back
        </button>

        <button
          className="agent-btn"
          onClick={() => navigate("/agent3")}
        >
          Next →
        </button>

      </div>

    </div>
  );
}