import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { runAortix } from "../api";

export default function Processing() {
  const navigate = useNavigate();
  const location = useLocation();
  const file = location.state?.file;

  useEffect(() => {
    if (!file) {
      alert("No dataset given");
      navigate("/upload");
      return;
    }

    let cancelled = false;

    async function process() {
      try {
        const result = await runAortix(file);
        if (!cancelled) {
          sessionStorage.setItem(
            "aortix_result",
            JSON.stringify(result)
          );
          navigate("/agent1")

        }
      } catch (err) {
        alert("Backend not available yet");
      }
    }

    process();

    // ✅ cleanup function
    return () => {
      cancelled = true;
    };
  }, [file, navigate]);

  return (
    <div style={{ textAlign: "center", marginTop: "100px" }}>
      <h2>Running Agentic AI…</h2>
    </div>
  );
}
