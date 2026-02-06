from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from agents.agent1_predictor import Agent1Predictor

app = FastAPI(title="AORTIX Backend")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AGENT 1 ----------------
agent1 = Agent1Predictor("models/heart_lstm.pt")

REQUIRED_COLS = [
    "heart_rate",
    "systolic_bp",
    "spo2",
    "ecg_risk",
    "pulmonary_pressure",
    "medication"
]

# ---------------- PIPELINE ----------------
@app.post("/run-aortix")
async def run_aortix(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    for col in REQUIRED_COLS:
        if col not in df.columns:
            return {"error": f"Missing column {col}"}

    numeric_cols = REQUIRED_COLS[:-1]

    row = df.iloc[0][numeric_cols].astype(float).values

    medication = 1.0 if str(df.iloc[0]["medication"]).lower() in ["yes","1","true"] else 0.0

    row = np.append(row, medication)

    sequence = np.tile(row, (10,1))

    agent1_output = agent1.predict(sequence)

    return {
        "agent1": agent1_output
    }
