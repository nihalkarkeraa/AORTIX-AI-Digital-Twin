import { useNavigate } from "react-router-dom";
import HeartView from "../components/HeartView";
import "./Agent.css";

export default function Agent1Prediction() {

  const navigate = useNavigate();

  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : {};

  const agentOutput = data?.agent1 || {};

  return (
    <div className="agent-container">

      {/* HEADER */}
      <div className="agent-header">
        <h2 className="glow-text">Agent 1 : Prediction Phase</h2>
        <span>AI Cardiovascular Digital Twin Analysis</span>
      </div>

      {/* STATS */}
      <div className="dashboard-grid">

        <div className="stat-card hover-glow">
          <h4>Prediction Score</h4>
          <p className="stat-value danger">
            {agentOutput.risk_score || "78%"}
          </p>
        </div>

        <div className="stat-card hover-glow">
          <h4>Status</h4>
          <p className="stat-value success">
            {agentOutput.status || "Active"}
          </p>
        </div>

        <div className="stat-card hover-glow">
          <h4>Confidence</h4>
          <p className="stat-value">
            {agentOutput.confidence || "92%"}
          </p>
        </div>

      </div>

      {/* MAIN CONTENT */}
      <div className="agent-content">

        {/* LEFT - HEART VISUAL */}
        <div className="agent-left glass-card">
          <h3 className="section-title">Heart Simulation</h3>
          <HeartView outputs={agentOutput} />
        </div>

        {/* RIGHT - ANALYSIS */}
        <div className="agent-right glass-card">
          <h3 className="section-title">AI Analysis</h3>

          {Object.keys(agentOutput).length > 0 ? (
            <div className="analysis-list">
              {Object.entries(agentOutput).map(([key, value]) => (
                <div className="analysis-item" key={key}>
                  <span className="analysis-key">
                    {key.replace("_", " ")}
                  </span>
                  <span className="analysis-value">
                    {value}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-data">
              No data available (Demo mode)
            </p>
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
          className="agent-btn primary"
          onClick={() => navigate("/agent2")}
        >
          Next →
        </button>

      </div>

    </div>
  );
}