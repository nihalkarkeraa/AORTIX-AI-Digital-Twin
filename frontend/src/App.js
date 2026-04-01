import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import "./App.css";

// PAGES
import Splash from "./pages/Splash";
import Upload from "./pages/Upload";
import Processing from "./pages/Processing";

import Agent1Prediction from "./pages/Agent1Prediction";
import Agent2Treatment from "./pages/Agent2Treatment";
import Agent3Risk from "./pages/Agent3Risk";
import Agent4Decision from "./pages/Agent4Decision";

import FinalReport from "./pages/FinalReport";

/* ================================
   PAGE WRAPPER (Smooth Transition)
================================ */
function AnimatedRoutes() {
  const location = useLocation();

  return (
    <div className="page-fade">
      <Routes location={location} key={location.pathname}>

        {/* MAIN FLOW */}
        <Route path="/" element={<Splash />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/processing" element={<Processing />} />

        {/* AGENTS */}
        <Route path="/agent1" element={<Agent1Prediction />} />
        <Route path="/agent2" element={<Agent2Treatment />} />
        <Route path="/agent3" element={<Agent3Risk />} />
        <Route path="/agent4" element={<Agent4Decision />} />

        {/* FINAL REPORT */}
        <Route path="/final" element={<FinalReport />} />

        {/* 404 PAGE */}
        <Route
          path="*"
          element={
            <div style={{
              height: "100vh",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              flexDirection: "column",
              background: "linear-gradient(135deg,#02080f,#071a29)",
              color: "#00c6ff"
            }}>
              <h1 style={{ fontSize: "3rem", marginBottom: "10px" }}>
                🚫 404
              </h1>
              <p>Page Not Found</p>
            </div>
          }
        />

      </Routes>
    </div>
  );
}

/* ================================
   MAIN APP
================================ */
export default function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <AnimatedRoutes />
      </BrowserRouter>
    </div>
  );
}