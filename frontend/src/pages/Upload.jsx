// pages/Upload.jsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Upload.css";

export default function Upload() {
  const [file, setFile] = useState(null);
  const nav = useNavigate();

  const submit = () => {
    if (!file) return alert("No dataset given");
    if (!file.name.endsWith(".csv"))
      return alert("Only CSV files allowed");

    localStorage.setItem("csv_uploaded", "true");
    nav("/processing", { state: { file } });
  };

  return (
    <div className="upload-container">
      <h1 className="upload-title">AORTIX</h1>

      <p className="upload-subtitle">
        Upload patient dataset to initialize the Cardiovascular Digital Twin
      </p>

      <div className="upload-box">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button className="upload-button" onClick={submit}>
          Submit Dataset
        </button>
      </div>
    </div>
  );
}
