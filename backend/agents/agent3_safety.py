import pandas as pd
import os

class Agent3Safety:

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        RULES_PATH = os.path.join(BASE_DIR, "agent3_rules.csv")

        self.rules = pd.read_csv(RULES_PATH)

    # -------- Risk Severity Index --------
    def compute_rsi(self, HR, BP, SPO2, ECG):
        rsi = (
            ECG * 0.4 +
            (BP / 180) * 0.3 +
            (HR / 150) * 0.2 +
            ((1 - SPO2/100) * 0.1)
        )
        return round(float(rsi), 2)

    # -------- Main Evaluation --------
    def evaluate(self, vitals, agent2_results):

        HR, BP, SPO2, ECG, PP = vitals

        # sort by score
        sorted_tx = sorted(agent2_results, key=lambda x: x["risk_reduction"], reverse=True)

        best = sorted_tx[0]
        second = sorted_tx[1]

        margin = round(best["risk_reduction"] - second["risk_reduction"], 3)
        confidence = round(min(1.0, margin * 5), 2)

        rsi = self.compute_rsi(HR, BP, SPO2, ECG)

        risk_level = "LOW"
        if rsi > 0.6:
            risk_level = "HIGH"
        elif rsi > 0.3:
            risk_level = "MEDIUM"

        status = "SAFE"
        if margin < 0.05:
            status = "WARNING"

        explanations = []

        for tx in sorted_tx:

            row = self.rules[self.rules["Treatment"] == tx["treatment"]].iloc[0]
            reason = row["Reason"]

            # simple medical checks
            if tx["treatment"] == "ACE" and BP < 100:
                reason = "Not recommended due to low blood pressure"

            if tx["treatment"] == "BETA" and HR < 60:
                reason = "Heart rate already low"

            explanations.append({
                "treatment": tx["treatment"],
                "score": round(tx["risk_reduction"], 3),
                "reason": reason,
                "recommended": tx["treatment"] == best["treatment"]
            })

        return {
            "status": status,
            "best_treatment": best["treatment"],
            "confidence": confidence,
            "margin": margin,
            "risk_index": rsi,
            "risk_level": risk_level,
            "treatments": explanations
        }
