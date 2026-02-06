import joblib
import pandas as pd

class Agent2TreatmentSimulator:
    def __init__(self):
        self.model = joblib.load("models/agent2_treatment_model.pkl")

        self.part_treatments = {
            "aorta": ["ACE_INHIBITOR", "ARB", "STATIN"],
            "left_ventricle": ["BETA_BLOCKER", "ACE_INHIBITOR", "AFTERLOAD_REDUCTION"],
            "right_ventricle": ["DIURETIC", "OXYGEN_THERAPY"],
            "pulmonary_artery": ["PULMONARY_VASODILATOR", "OXYGEN_THERAPY"],
            "pulmonary_veins": ["DIURETIC", "FLUID_RESTRICTION"],
            "mitral_valve": ["DIURETIC", "RATE_CONTROL"],
            "tricuspid_valve": ["DIURETIC", "VOLUME_MANAGEMENT"],
            "inferior_vena_cava": ["VOLUME_MANAGEMENT", "DIURETIC"]
        }

    import pandas as pd

def _predict(self, risk, csv_row, treatment):
    X = pd.DataFrame([{
        "risk_score": risk,
        "systolic_bp": csv_row["systolic_bp"],
        "heart_rate": csv_row["heart_rate"],
        "spo2": csv_row["spo2"],
        "ecg_risk": csv_row["ecg_risk"],
        "treatment": treatment
    }])

    prob = self.model.predict_proba(X)[0][1]
    return round(prob * 100, 1)


    def simulate(self, agent1_output, csv_row):
        results = {}

        for part, text in agent1_output.items():
            risk = float(text.split(":")[-1])
            results[part] = []

            for treatment in self.part_treatments[part]:
                confidence = self._predict(risk, csv_row, treatment)
                results[part].append({
                    "treatment": treatment.replace("_", " "),
                    "confidence": f"{confidence}%"
                })

        return results
