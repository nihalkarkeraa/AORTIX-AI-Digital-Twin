import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./Splash.css";

export default function Splash() {
  const navigate = useNavigate();

  useEffect(() => {
    const timerId = setTimeout(() => {
      navigate("/upload");
    }, 5000);

    return () => clearTimeout(timerId);
  }, [navigate]);

  return (
    <div className="splash-container">
      <h1 className="splash-title">AORTIX</h1>

      <p className="splash-subtitle">
        An AI-Driven Cardiovascular Digital Twin with Agentic AI and
        Generative AI for Treatment Simulation and Risk Prediction
      </p>

      <div className="splash-loader">
        Initializing Cardiovascular Digital Twin...
      </div>
    </div>
  );
  

}
