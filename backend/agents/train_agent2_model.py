import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

CSV_PATH = r"C:\Users\asus\aortix\backend\agent 2 traning data.csv"

EPOCHS = 80

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH)

encoder = LabelEncoder()
df["Treatment_Code"] = encoder.fit_transform(df["Treatment"])

X = df[["HR","BP","SPO2","ECG","PP","Treatment_Code"]].values
Y = df[["Risk_Reduction"]].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler,"backend/models/agent2_scaler.save")
joblib.dump(encoder,"backend/models/agent2_encoder.save")

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.float32)

# ---------------- MODEL ----------------
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

model = Agent2MLP()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.MSELoss()

# ---------------- TRAIN ----------------
print("\nTraining Agent-2...\n")

for epoch in range(EPOCHS):
    preds = model(X_train)
    loss = loss_fn(preds,Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10==0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

torch.save(model.state_dict(),"backend/models/agent2_mlp.pt")

print("\nAgent-2 model saved:")
print("backend/models/agent2_mlp.pt")
