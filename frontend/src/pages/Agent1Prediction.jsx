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
          onClick={() => navigate("/dataset")}
        >
          ← Back to Dataset
        </button>
      </div>
    );
  }

  const agentOutput = data.agent1;

  return (
    <div className="agent-container">
      {/* HEADER */}
      <div className="agent-header">
        <h2>Agent 1 : Prediction Phase</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* CONTENT */}
      <div className="agent-content">
        {/* LEFT SIDE — HEART */}
        <div className="agent-left">
          <HeartView outputs={agentOutput} />
        </div>

        {/* RIGHT SIDE — OUTPUT */}
        <div className="agent-right">
          <h3>Component-wise Analysis</h3>
          <div>
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
      </div>

      {/* CONTROLS */}
      <div className="agent-controls">
        <button
          className="agent-btn secondary"
          onClick={() => navigate("/dataset")}
        >
          ← Back
        </button>

        <button
          className="agent-btn"
          onClick={() => navigate("/agent2")
}
        >
          Next →
        </button>
      </div>
    </div>
  );
}
