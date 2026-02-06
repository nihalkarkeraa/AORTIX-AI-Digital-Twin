import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ---------------- CONFIG ----------------
SEQ_LEN = 10
BATCH_SIZE = 32
EPOCHS = 40
LR = 0.001

INPUT_COLS = [
    "heart_rate",
    "systolic_bp",
    "spo2",
    "ecg_risk",
    "pulmonary_pressure",
    "medication"
]

TARGET_COLS = [
    "aorta_risk",
    "left_ventricle_risk",
    "right_ventricle_risk",
    "pulmonary_artery_risk",
    "pulmonary_veins_risk",
    "mitral_valve_risk",
    "tricuspid_valve_risk",
    "ivc_risk"
]

# ---------------- DATASET ----------------
class HeartDataset(Dataset):
    def __init__(self, df):
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        X = self.x_scaler.fit_transform(df[INPUT_COLS])
        Y = self.y_scaler.fit_transform(df[TARGET_COLS])

        self.X_seq = []
        self.Y_seq = []

        for i in range(len(df) - SEQ_LEN):
            self.X_seq.append(X[i:i + SEQ_LEN])
            self.Y_seq.append(Y[i + SEQ_LEN])

        self.X_seq = torch.tensor(self.X_seq, dtype=torch.float32)
        self.Y_seq = torch.tensor(self.Y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.Y_seq[idx]

# ---------------- MODEL ----------------
class HeartLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(INPUT_COLS),
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64, len(TARGET_COLS))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---------------- TRAINING ----------------
def train():
    df = pd.read_csv("heart_training_data.csv")

    dataset = HeartDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HeartLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for X, Y in loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/heart_lstm.pt")
    print(" heart_lstm.pt saved in models/")

if __name__ == "__main__":
    train()
