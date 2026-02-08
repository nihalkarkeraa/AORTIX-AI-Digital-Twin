from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from agents.agent1_predictor import Agent1Predictor
from agents.agent2_treatment import Agent2TreatmentSimulator
from agents.agent3_safety import Agent3Safety
from agents.agent4_decision import Agent4Decision

# NEW — GenAI Agent
from genai.agent5_summary import Agent5Summary

app = FastAPI(title="AORTIX Backend")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- AGENTS ----------------
agent1 = Agent1Predictor("models/heart_lstm.pt")
agent2 = Agent2TreatmentSimulator()
agent3 = Agent3Safety()
agent4 = Agent4Decision()

# NEW
agent5 = Agent5Summary()

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

    # ---------------- AGENT 1 INPUT ----------------
    numeric_cols = REQUIRED_COLS[:-1]

    row = df.iloc[0][numeric_cols].astype(float).values

    medication = 1.0 if str(df.iloc[0]["medication"]).lower() in ["yes","1","true"] else 0.0

    row = np.append(row, medication)

    sequence = np.tile(row, (10,1))

    # ---------------- AGENT 1 ----------------
    agent1_output = agent1.predict(sequence)

    # ---------------- AGENT 2 ----------------
    vitals = [
        float(df.iloc[0]["heart_rate"]),
        float(df.iloc[0]["systolic_bp"]),
        float(df.iloc[0]["spo2"]),
        float(df.iloc[0]["ecg_risk"]),
        float(df.iloc[0]["pulmonary_pressure"])
    ]

    agent2_output = agent2.simulate(vitals=vitals)

    # ---------------- AGENT 3 ----------------
    agent3_output = agent3.evaluate(
        vitals=vitals,
        agent2_results=agent2_output["all"]
    )

    # ---------------- AGENT 4 ----------------
    agent4_output = agent4.decide(agent3_output)

    # ---------------- AGENT 5 (GEN AI) ----------------
    gen_ai_report = agent5.generate(
        agent1_output,
        agent2_output,
        agent3_output,
        agent4_output
    )

    return {
        "agent1": agent1_output,
        "agent2": agent2_output,
        "agent3": agent3_output,
        "agent4": agent4_output,
        "gen_ai_report": gen_ai_report
    }
