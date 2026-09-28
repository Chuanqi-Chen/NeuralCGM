# Stochastic Lorenz 96 System (40 States) with Noisy Observation

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import RMSE

mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

####################################################
################# Data Preparation #################
####################################################

u = torch.tensor(np.load("data/SimulationData/L96_Data_NoisyObs.npy"), dtype=torch.float).to(device)

# Train /Test
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]

# Normalization
class Normalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)
        self.eps = eps

    def encode(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / (std + self.eps)

    def decode(self, x):
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean

Normalizer = Normalizer(train_u)
train_u_normalized = Normalizer.encode(train_u)
test_u_normalized = Normalizer.encode(test_u)

################################################################
################# Model Architecture  ##########################
################################################################

################# MLP #################
class StateTransitionMLP(nn.Module):
    def __init__(self, state_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, state_dim))
    def forward(self, x):
        out = self.net(x)
        return out


################# CNN #################
class StateTransitionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(1, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Conv1d(32, 1, 3, 1, 1))
    def forward(self, x):
        # x: (N, S)
        out = self.net(x[:, None])[:, 0]
        return out


######################################################
################# Training  ##########################
######################################################

epochs = 1000
batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u_normalized[:-1], train_u_normalized[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history = []

model = StateTransitionMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)

        u_pred = model(u)
        loss = nnF.mse_loss(u_next, u_pred)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    train_loss /= train_num_batches
    train_loss_history.append(train_loss)

    end_time = time.time()
    print(f"ep {ep} | "
          f"time: {end_time - start_time:.4f} | "
          f"loss: {train_loss:.4f} | ")
"""
# torch.save(model.state_dict(), r"./model/L96_DNN.pt")
# np.save(r"./model/L96_DNN_train_loss_history.npy", train_loss_history)

model.load_state_dict(torch.load(r"./model/L96_DNN.pt"))


#################################################
################# Model Inference ###############
#################################################

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L96_Data.npy"), dtype=torch.float).to(device)
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]

# State Forecast
with torch.no_grad():
    train_u_normalized_pred = model(train_u_normalized)
train_u_pred = Normalizer.decode(train_u_normalized_pred)
train_forecast_RMSE = RMSE(train_u_true[1:], train_u_pred[:-1])
print("train forecast RMSE: ", train_forecast_RMSE)
with torch.no_grad():
    test_u_normalized_pred = model(test_u_normalized)
test_u_pred = Normalizer.decode(test_u_normalized_pred)
test_forecast_RMSE = RMSE(test_u_true[1:], test_u_pred[:-1])
print("test forecast RMSE: ", test_forecast_RMSE)

param_count = sum(p.numel() for p in model.parameters())
print("param count: ", param_count)


#####################################################
################# Visualization Data ################
#####################################################

# Multiple-Step State Forecasting
def forecast_next_state(u_normalized):
    with torch.no_grad():
        u_normalized_pred = model(u_normalized)
    return u_normalized_pred
pred_step = 100
test_u_normalized_multistep_pred = torch.zeros(Ntest, pred_step+1, state_dim, device=device)
test_u_normalized_multistep_pred[:, 0] = test_u_normalized
for n in range(pred_step):
    print("Forecast Step:", n+1)
    test_u_normalized_multistep_pred[:, n+1] = forecast_next_state(test_u_normalized_multistep_pred[:, n])
test_u_multistep_pred = Normalizer.decode(test_u_normalized_multistep_pred)
# np.save(r"./data/VisualizationData/L96_Forecast_DNN.npy", test_u_multistep_pred)

memory = pred_step + 1
test_u_multistep_pred = test_u_multistep_pred[:-memory+1]
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1)
RMSE(test_u_multistep_pred, test_rollout_true)


