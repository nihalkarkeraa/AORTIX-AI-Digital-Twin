import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Upload.css";

export default function Upload() {
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Idle");
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    setFile(selectedFile);
    setSuccess(false);
    setError("");
    setProgress(0);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const processDataset = async () => {
    if (!file) {
      setError("Please select a CSV file.");
      return;
    }

    try {
      setStatus("Connecting to backend...");
      setProgress(30);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/run-aortix", {
        method: "POST",
        body: formData,
      });

      setProgress(70);

      if (!response.ok) {
        throw new Error("Backend failed");
      }

      const result = await response.json();

      setProgress(100);
      setStatus("Completed");
      setSuccess(true);

      // Store backend result
      sessionStorage.setItem("aortix_result", JSON.stringify(result));

      // Navigate to Agent1
      setTimeout(() => {
        navigate("/agent1");
      }, 800);

    } catch (err) {
      console.error(err);
      setStatus("Error");
      setError("Backend error. Check if backend is running.");
    }
  };

  return (
    <div className="upload-container">
      <h1 className="upload-title">Upload Dataset</h1>

      <p className="upload-subtitle">
        Upload patient CSV file for AI cardiovascular analysis
      </p>

      <div
        className={`upload-box ${dragActive ? "drag-active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {file && (
          <div className="file-preview">
            📄 {file.name}
          </div>
        )}

        {progress > 0 && (
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        )}

        {success && (
          <div className="success-check">
            ✔ Upload Successful
          </div>
        )}

        {error && (
          <div style={{ color: "red", marginTop: "10px" }}>
            {error}
          </div>
        )}

        <button className="upload-button" onClick={processDataset}>
          Process Dataset
        </button>

        <div className="status-indicator">
          Backend Status: {status}
        </div>
      </div>
    </div>
  );
}
