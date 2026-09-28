# Kuramoto–Sivashinsky Equation (KSE)
# Observed States: 2 points uniformly distributed in the spatial domain with observation noise
# Unobserved States: spatial domain

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import math
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

# Data Resolution: dt=0.25, dx=22/2048
u = np.load("data/SimulationData/KSE_Data_NoisyObs.npy")
# u_true = np.load("data/KSE_Data.npy")
# sigma_obs = 0.15*np.std(u_true)
sigma_obs = 0.17799336014506514


# Sub-Sampling: dt=1, dx=22/128
u = u[::4, ::16]  # (50000, 128)
dt_obs = 1.

# Spatial Domain
Lx = 22
gridx = np.arange(0, Lx, 22/2048)
gridx = gridx[::16]

# Train/Test & State/Observation
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
indices_u1 = np.arange(0, 128, 64) # 2 points
indices_u2 = np.arange(0, 128)[np.arange(0, 128)%64!=0] # 126 points
obs_dim = len(indices_u1)
train_state = torch.tensor(u[:Ntrain], dtype=torch.float)
test_state = torch.tensor(u[-Ntest:], dtype=torch.float)
train_obs = torch.tensor(u[:Ntrain], dtype=torch.float)[:, indices_u1]
test_obs = torch.tensor(u[-Ntest:], dtype=torch.float)[:, indices_u1]
grid_state = gridx
grid_obs = gridx[indices_u1]



########################################################
##################### Interpolation ####################
########################################################

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/KSE_Data.npy"), dtype=torch.float)
u_true = u_true[::4, ::16]
train_state_true = u_true[:Ntrain]
test_state_true = u_true[-Ntest:]
train_obs_true = u_true[:Ntrain, indices_u1]
test_obs_true = u_true[-Ntest:, indices_u1]


test_obs_periodic = torch.cat([test_obs, test_obs[:, :1]], dim=1)
test_interpolation = nnF.interpolate(test_obs_periodic.unsqueeze(1), size=129, mode="linear", align_corners=True).squeeze(1)[:, :-1]
test_interpolation_RMSE = RMSE(test_state_true[:, indices_u2], test_interpolation[:, indices_u2])
print("test Periodic interpolation RMSE:", test_interpolation_RMSE)

# np.save(r"./data/VisualizationData/KSE_Interpolation.npy", test_interpolation)


