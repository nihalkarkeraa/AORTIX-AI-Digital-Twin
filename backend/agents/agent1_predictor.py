import torch
import torch.nn as nn
import numpy as np

HEART_PARTS = [
    "aorta",
    "left_ventricle",
    "right_ventricle",
    "pulmonary_artery",
    "pulmonary_veins",
    "mitral_valve",
    "tricuspid_valve",
    "inferior_vena_cava"
]

HEART_PART_LABELS = {
    "aorta": "Aorta",
    "left_ventricle": "Left Ventricle",
    "right_ventricle": "Right Ventricle",
    "pulmonary_artery": "Pulmonary Artery",
    "pulmonary_veins": "Pulmonary Veins",
    "mitral_valve": "Mitral Valve",
    "tricuspid_valve": "Tricuspid Valve",
    "inferior_vena_cava": "Inferior Vena Cava"
}

class HeartLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6,64,2,batch_first=True)
        self.fc = nn.Linear(64,8)

    def forward(self,x):
        o,_ = self.lstm(x)
        return torch.sigmoid(self.fc(o[:,-1,:]))

class Agent1Predictor:

    def __init__(self, path):
        self.model = HeartLSTM()
        self.model.load_state_dict(torch.load(path,map_location="cpu"))
        self.model.eval()

    def predict(self, sequence):

        x = torch.tensor(sequence,dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            preds = self.model(x)[0].numpy()

        output = {}

        for i,part in enumerate(HEART_PARTS):

            risk = float(preds[i])

            if risk < 0.3:
                status = "Low Risk"
            elif risk < 0.6:
                status = "Moderate Risk"
            else:
                status = "High Risk"

            output[part] = f"{HEART_PART_LABELS[part]} → {status} : {round(risk,2)}"

        return output
