import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Upload.css";
import heart from "../assets/heart_black_boxes.png";

export default function Upload() {

  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Idle");
  const [success, setSuccess] = useState(false);

  /* ===============================
     HANDLE FILE SELECT
  =============================== */
  const handleFile = (selectedFile) => {
    console.log("FILE SELECTED:", selectedFile); // DEBUG
    setFile(selectedFile);
    setProgress(0);
    setSuccess(false);
  };

  /* ===============================
     PROCESS DATASET
  =============================== */
  const processDataset = async () => {

    setStatus("Processing...");
    setProgress(40);

    try {

      // OPTIONAL BACKEND CALL
      if (file) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://127.0.0.1:8000/run-aortix", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          sessionStorage.setItem("aortix_result", JSON.stringify(result));
        }
      }

      setProgress(100);
      setSuccess(true);
      setStatus("Completed");

    } catch (err) {
      console.log("Backend skipped (demo mode)");
    }

    // ALWAYS GO NEXT
    setTimeout(() => {
      navigate("/agent1");
    }, 800);
  };

  return (
    <div className="upload-container">

      {/* LEFT HEART */}
      <div className="heart-panel">
        <img src={heart} alt="Heart" />
      </div>

      {/* RIGHT PANEL */}
      <div className="upload-panel">

        <h1 className="upload-title">AORTIX</h1>

        <p className="upload-subtitle">
          AI Cardiovascular Digital Twin
        </p>

        {/* FILE UPLOAD */}
        <div className="upload-box">

          {/* CUSTOM FILE BUTTON */}
          <label className="file-upload">
            <input
              type="file"
              accept=".csv"
              onChange={(e) => handleFile(e.target.files[0])}
              hidden
            />
            📂 Choose File
          </label>

          {/* FILE PREVIEW */}
          {file && (
            <div className="file-preview">
              ✅ {file.name}
            </div>
          )}

          {/* PROGRESS */}
          {progress > 0 && (
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          )}

          {/* SUCCESS */}
          {success && (
            <div className="success-check">
              ✔ Ready to Proceed
            </div>
          )}

          {/* BUTTON */}
          <button
            className="upload-button"
            onClick={processDataset}
          >
            Initialize Digital Twin
          </button>

          {/* STATUS */}
          <div className="status-indicator">
            Status: {status}
          </div>

        </div>

      </div>

    </div>
  );
}