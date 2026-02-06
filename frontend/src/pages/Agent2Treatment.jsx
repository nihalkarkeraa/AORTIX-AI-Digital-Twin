import { useNavigate } from "react-router-dom";
import "./Agent.css";

export default function Agent2Treatment() {
  const navigate = useNavigate();

  const raw = sessionStorage.getItem("aortix_result");
  const data = raw ? JSON.parse(raw) : null;

  // SAFETY GUARD
  if (!data || !data.agent2) {
    return (
      <div className="agent-container">
        <h2>No Agent 2 data available</h2>

        <div className="agent-controls">
          <button
            className="agent-btn secondary"
            onClick={() => navigate("/agent1")}
          >
            ← Back to Agent 1
          </button>
        </div>
      </div>
    );
  }

  const agent2Output = data.agent2;

  return (
    <div className="agent-container">
      {/* HEADER */}
      <div className="agent-header">
        <h2>Agent 2 : Treatment Simulation Phase</h2>
        <span>AORTIX — Cardiovascular Digital Twin</span>
      </div>

      {/* CONTENT */}
      <div className="agent-content">
        {/* LEFT — SIMULATION VISUAL PLACEHOLDER */}
        <div className="agent-left">
          <div
            style={{
              width: "280px",
              height: "280px",
              borderRadius: "16px",
              border: "2px dashed rgba(255,255,255,0.6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.1rem",
              opacity: 0.85,
              textAlign: "center"
            }}
          >
            Treatment<br />Simulation
          </div>
        </div>

        {/* RIGHT — TREATMENT OUTPUT */}
        <div className="agent-right">
          <h3>Part-wise Treatment Plan</h3>

          {Object.entries(agent2Output).map(([part, treatments]) => (
            <div key={part} style={{ marginBottom: "15px" }}>
              <b style={{ textTransform: "capitalize" }}>
                {part.replaceAll("_", " ")}
              </b>

              <ul style={{ marginTop: "6px" }}>
                {treatments.map((t, idx) => (
                  <li key={idx}>
                    {t.treatment} — <b>{t.confidence}</b>
                  </li>
                ))}
              </ul>
            </div>
          ))}
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
