// pages/FinalReport.jsx
import "./FinalReport.css";

export default function FinalReport() {
  const data = JSON.parse(sessionStorage.getItem("aortix_result"));

  return (
    <div className="final-container">
  <div className="final-card">

    <h1 className="final-title">AORTIX REPORT</h1>

    {/* METRIC CARDS */}
    <div className="metrics-grid">
      <div className="metric-card">
        <h3>Risk Score</h3>
        <p className="metric-value high">78%</p>
      </div>

      <div className="metric-card">
        <h3>Blood Pressure</h3>
        <p className="metric-value">120 / 80</p>
      </div>

      <div className="metric-card">
        <h3>Heart Rate</h3>
        <p className="metric-value">78 bpm</p>
      </div>
    </div>

    {/* ECG ANIMATION */}
    <div className="ecg-container">
      <div className="ecg-line"></div>
    </div>

    {/* AI CHAT */}
    <div className="chat-wrapper">
      <div className="chat-bubble typing">
        The cardiovascular profile indicates elevated risk factors...
      </div>
    </div>

    {/* DOWNLOAD BUTTON */}
    <div className="download-section">
      <button className="download-btn">Download PDF</button>
    </div>

  </div>
</div>

  );
}
