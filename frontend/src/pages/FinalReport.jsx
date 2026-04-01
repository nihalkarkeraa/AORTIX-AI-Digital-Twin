import "./FinalReport.css";
import html2pdf from "html2pdf.js";

export default function FinalReport() {

  /* ===============================
     SAFE DATA FETCH
  =============================== */
  const stored = sessionStorage.getItem("aortix_result");
  const data = stored ? JSON.parse(stored) : {};

  /* ===============================
     PDF DOWNLOAD
  =============================== */
const downloadPDF = () => {

  const element = document.querySelector(".final-card");

  if (!element) {
    alert("Report not ready yet!");
    return;
  }

  const opt = {
    margin: 0.5,
    filename: "AORTIX_Report.pdf",
    image: { type: "jpeg", quality: 1 },
    html2canvas: {
      scale: 3,
      useCORS: true,
      logging: false
    },
    jsPDF: {
      unit: "in",
      format: "a4",
      orientation: "portrait"
    }
  };

  // 🔥 Delay ensures DOM fully rendered
  setTimeout(() => {
    html2pdf().from(element).set(opt).save();
  }, 500);
};
  /* ===============================
     DEFAULT DEMO DATA (IMPORTANT)
  =============================== */
  const riskScore = data?.risk_score || "78";
  const bp = data?.bp || "120 / 80";
  const heartRate = data?.heart_rate || "78 bpm";
  const summary =
    data?.summary ||
    "AI analysis suggests moderate cardiovascular risk. Lifestyle improvements, regular monitoring, and preventive care are recommended.";

  return (

    <div className="final-container">

      <div className="final-card">

        {/* TITLE */}
        <h1 className="final-title glow">
          AORTIX AI REPORT
        </h1>

        {/* ===============================
           METRICS
        =============================== */}
        <div className="metrics-grid">

          <div className="metric-card">
            <h3>Risk Score</h3>
            <p className="metric-value high">
              {riskScore}%
            </p>
          </div>

          <div className="metric-card">
            <h3>Blood Pressure</h3>
            <p className="metric-value">
              {bp}
            </p>
          </div>

          <div className="metric-card">
            <h3>Heart Rate</h3>
            <p className="metric-value">
              {heartRate}
            </p>
          </div>

        </div>

        {/* ===============================
           ECG ANIMATION
        =============================== */}
        <div className="ecg-container">
          <div className="ecg-line"></div>
        </div>

        {/* ===============================
           AI SUMMARY
        =============================== */}
        <div className="chat-wrapper">
          <div className="chat-bubble">
            {summary}
          </div>
        </div>

        {/* ===============================
           DOWNLOAD BUTTON
        =============================== */}
        <div className="download-section">
          <button
            className="download-btn"
            onClick={downloadPDF}
          >
            ⬇ Download Full Report
          </button>
        </div>

      </div>

    </div>
  );
}