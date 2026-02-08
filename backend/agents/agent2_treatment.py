import torch
import torch.nn as nn
import numpy as np
import joblib

TREATMENTS = ["ACE", "BETA", "ARB"]

# ---------------- MLP MODEL ----------------
class Agent2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self,x):
        return self.net(x)

# ---------------- AGENT 2 ----------------
class Agent2TreatmentSimulator:

    def __init__(self):
        self.model = Agent2MLP()
        torch.load("models/agent2_mlp.pt", map_location="cpu")

        self.model.eval()

        # same scaler + label encoder you used during training
        self.scaler = joblib.load("models/agent2_scaler.save")
        self.encoder = joblib.load("models/agent2_encoder.save")


    def simulate(self, vitals):
        """
        vitals = [HR, BP, SPO2, ECG, PP]
        """

        results = []

        for t in TREATMENTS:
            code = self.encoder.transform([t])[0]

            row = vitals + [code]
            row = self.scaler.transform([row])

            x = torch.tensor(row, dtype=torch.float32)

            with torch.no_grad():
                score = float(self.model(x).item())

            results.append({
                "treatment": t,
                "risk_reduction": round(score, 3)
            })

        # sort by risk reduction
        results = sorted(results, key=lambda x: x["risk_reduction"], reverse=True)

        return {
            "best": results[0],
            "worst": results[-1],
            "all": results
        }
