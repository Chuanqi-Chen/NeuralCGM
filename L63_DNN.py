# Stochastic Lorenz 63 System with Noisy Observations

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

u = torch.tensor(np.load("data/SimulationData/L63_Data_NoisyObs.npy"), dtype=torch.float)

# Train /Test
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]


###################################################################
################# DNN: State Transition Operator  #################
###################################################################

class StateTransitionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(3, 32), nn.SiLU(),
                                 nn.Linear(32, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)

##############################################
################# Train DNN  #################
##############################################

epochs = 500
batch_size = 1000
train_tensor = torch.utils.data.TensorDataset(train_u[:-1], train_u[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history = []

model = StateTransitionNet().to(device)
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
# torch.save(model.state_dict(), r"./model/L63_DNN.pt")
# np.save(r"./model/L63_DNN_train_loss_history.npy", train_loss_history)

model.load_state_dict(torch.load(r"model/L63_DNN.pt"))


#################################################
################# Model Inference ###############
#################################################

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L63_Data.npy"), dtype=torch.float32)
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]

with torch.no_grad():
    train_u_pred = model(train_u)
train_forecast_RMES = RMSE(train_u_true[1:], train_u_pred[:-1])
print("train forecast RME: ", train_forecast_RMES)

with torch.no_grad():
    test_u_pred = model(test_u)
test_forecast_RMES = RMSE(test_u_true[1:], test_u_pred[:-1])
print("test forecast RME: ", test_forecast_RMES)

num_params = sum(p.numel() for p in model.parameters())
print("param count: ", num_params)


#####################################################
################# Visualization Data ################
#####################################################

# Multiple-Step State Forecasting
def forecast_next_state(u):
    with torch.no_grad():
        u_pred = model(u)
    return u_pred
pred_step = 100
test_u_multistep_pred = torch.zeros(Ntest, pred_step+1, state_dim, device=device)
test_u_multistep_pred[:, 0] = test_u
for n in range(pred_step):
    print("Forecast Step:", n+1)
    test_u_multistep_pred[:, n+1] = forecast_next_state(test_u_multistep_pred[:, n])


memory = pred_step+1
test_u_multistep_pred = test_u_multistep_pred[:-memory+1]
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1)
RMSE(test_u_multistep_pred, test_rollout_true)
# np.save(r"./data/VisualizationData/L63_Forecast_DNN.npy", test_u_multistep_pred)
