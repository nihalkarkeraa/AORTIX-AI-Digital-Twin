import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\asus\Desktop\backend\agent2_treatment_data.csv")


X = df.drop("effective", axis=1)
y = df["effective"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(), ["treatment"]),
    ("num", "passthrough",
     ["risk_score", "systolic_bp", "heart_rate", "spo2", "ecg_risk"])
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X, y)

joblib.dump(model, "models/agent2_treatment_model.pkl")

print("Agent 2 ML model trained and saved")
