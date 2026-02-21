import { BrowserRouter, Routes, Route } from "react-router-dom";

import Splash from "./pages/Splash";
import Upload from "./pages/Upload";
import Processing from "./pages/Processing";

import Agent1Prediction from "./pages/Agent1Prediction";
import Agent2Treatment from "./pages/Agent2Treatment";
import Agent3Risk from "./pages/Agent3Risk";
import Agent4Decision from "./pages/Agent4Decision";

import FinalReport from "./pages/FinalReport";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>

        <Route path="/" element={<Splash />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/processing" element={<Processing />} />

        <Route path="/agent1" element={<Agent1Prediction />} />
        <Route path="/agent2" element={<Agent2Treatment />} />
        <Route path="/agent3" element={<Agent3Risk />} />
        <Route path="/agent4" element={<Agent4Decision />} />

        <Route path="/final" element={<FinalReport />} />

        <Route path="*" element={<h2>404</h2>} />

      </Routes>
    </BrowserRouter>
  );
}
