import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent2Treatment() {

  const navigate = useNavigate();

  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : null;

  if (!data || !data.agent2) {
    return (
      <div className="agent-container">
        <h2>No Agent-2 data available</h2>
        <button className="agent-btn" onClick={() => navigate("/agent1")}>
          ← Back
        </button>
      </div>
    );
  }

  const { best, worst } = data.agent2;

  return (
    <div className="agent-container">

      {/* HEADER */}
      <div className="agent-header">
        <h2>Agent 2 : Treatment Simulation</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* CONTENT */}
      <div className="agent-content">

        {/* LEFT — WORST */}
        <div className="agent-left">
          <h3>Worst Medication</h3>
          <p>{worst.treatment}</p>
          <p>Score: {worst.risk_reduction}</p>
        </div>

        {/* RIGHT — BEST */}
        <div className="agent-right">
          <h3>Best Medication</h3>
          <p>{best.treatment}</p>
          <p>Score: {best.risk_reduction}</p>
        </div>

      </div>

      {/* CONTROLS */}
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
