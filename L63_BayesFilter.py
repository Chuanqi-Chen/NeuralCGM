# Stochastic Lorenz 63 System with Noisy Observation
# System States: [x, y, z]
# Observations: Noisy x

import numpy as np
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from BayesFilter import *
from metrics import RMSE, NLL



mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

device = "cpu"

####################################################
################# Data Preparation #################
####################################################

u = torch.tensor(np.load("data/SimulationData/L63_Data_NoisyObs.npy"), dtype=torch.float)

sigma_obs = 2.  # Measurement Noise
sigma_dyn = 0.5
dt_obs = 0.05

# Train /Test
Nt_obs = len(u)
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]

# State /  Obs
state_dim = 3
obs_dim = 1
indices_u1 = [0]
indices_u2 = [1, 2]
train_state = train_u
train_obs = train_u[:, :1]
test_state = test_u
test_obs = test_u[:, :1]


##########################################################
################# State-Space Model ######################
##########################################################

def dynamics_L63(u):
    # u shape (N, state_dim)
    sigma, rho, beta = 10, 28, 8/3
    u_dot = torch.zeros_like(u)
    u_dot[:, 0] = sigma * (u[:, 1] - u[:, 0])
    u_dot[:, 1] = u[:, 0] * (rho-u[:, 2]) - u[:, 1]
    u_dot[:, 2] = u[:, 0] * u[:, 1] - beta * u[:, 2]
    return u_dot

def forecast_L63(u, delta_t=dt_obs):
    # u shape (N, state_dim)
    sigma_dyn = 0.5
    dt_sim = 0.001 # Internal Integration Time Step
    Nt_sim = int(delta_t/dt_sim)
    for _ in range(Nt_sim):
        u = u + dynamics_L63(u) * dt_sim + sigma_dyn * dt_sim**0.5 * torch.randn_like(u)
    return u

def obs_map_L63(u):
    # u shape (N, state_dim)
    return u[:, :1]

def Jacobian_dyn_L63(x):
    # x: (B, 3)
    sigma, rho, beta = 10, 28, 8 / 3
    B = x.shape[0]

    J = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)

    J[:, 0, 0] = -sigma
    J[:, 0, 1] =  sigma

    J[:, 1, 0] =  rho - x[:, 2]
    J[:, 1, 1] = -1.0
    J[:, 1, 2] = -x[:, 0]

    J[:, 2, 0] =  x[:, 1]
    J[:, 2, 1] =  x[:, 0]
    J[:, 2, 2] = -beta

    return J

def Jacobian_obs_L63(x):
    # x: (B, 3)
    B = x.shape[0]

    H = torch.zeros(B, 1, 3, device=x.device, dtype=x.dtype)
    H[:, 0, 0] = 1.0

    return H

def metrics_unobs(x, mu, cov):
    #  metrics on unobserved states
    u2_true = x[:, indices_u2]
    mu_DA_u2 = mu[:, indices_u2]
    if cov.dim() == 2:
        cov_DA_u2 = cov[:, indices_u2]
    else:
        cov_DA_u2 = cov.diagonal(dim1=1, dim2=2)[:, indices_u2]
    rmse = RMSE(u2_true, mu_DA_u2)
    nll = NLL(u2_true, mu_DA_u2, cov_DA_u2)
    return {"RMSE": rmse, "NLL" : nll}


#############################################
################# Benchmark #################
#############################################

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L63_Data.npy"), dtype=torch.float32)
train_state_true = u_true[:Ntrain]
test_state_true = u_true[-Ntest:]
train_obs_true = u_true[:Ntrain, :1]
test_obs_true = u_true[-Ntest:, :1]

# EKF
torch.manual_seed(0)
np.random.seed(0)
mu0 = train_state.mean(dim=0)
cov0 = torch.cov(train_state.T)
st = time.time()
mu_EKF, cov_EKF = EKF(y=test_obs, state_dim=state_dim, forecast_dyn=dynamics_L63, obs_map=obs_map_L63, cov_dyn=sigma_dyn**2 * torch.eye(state_dim), cov_obs=sigma_obs ** 2 * torch.eye(obs_dim), dt_obs=dt_obs, dt_sim=0.001, Jacobian_dyn=Jacobian_dyn_L63, Jacobian_obs=Jacobian_obs_L63, mu0=mu0, cov0=cov0)
et = time.time()
print("EKF metrics: ", metrics_unobs(test_state_true, mu_EKF, cov_EKF))
print("EKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L63_DA_EKF.npz", mean=mu_EKF, std=cov_EKF.diagonal(dim1=-2, dim2=-1)**0.5)



# UKF
torch.manual_seed(0)
np.random.seed(0)
mu0 = train_state.mean(dim=0)
cov0 = torch.cov(train_state.T)
st = time.time()
mu_UKF, cov_UKF = UKF(y=test_obs, state_dim=state_dim, forecast_dyn=dynamics_L63, obs_map=obs_map_L63, cov_dyn=sigma_dyn**2*torch.eye(state_dim), cov_obs=sigma_obs ** 2 * torch.eye(obs_dim), dt_obs=0.05, dt_sim=0.001, alpha=0.3, k=0, beta=2, mu0=mu0, cov0=cov0)
et = time.time()
print("UKF metrics: ", metrics_unobs(test_state_true, mu_UKF, cov_UKF))
print("UKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L63_DA_UKF.npz", mean=mu_UKF, std=cov_UKF.diagonal(dim1=-2, dim2=-1)**0.5)



# EnKF
torch.manual_seed(0)
np.random.seed(0)
ens_size = 200
x0 = train_state[np.random.randint(0, Ntrain, size=ens_size)]
st = time.time()
x_ens = EnKF(y=test_obs, state_dim=state_dim, forecast=forecast_L63, obs_map=obs_map_L63, cov_obs=sigma_obs**2*torch.eye(obs_dim), J=ens_size, x0=x0)
mu_EnKF = x_ens.mean(dim=0)
# cov_EnKF = torch.einsum("nti,ntj->tij", x_ens-mu_EnKF, x_ens-mu_EnKF) / (x_ens.shape[0]-1)
cov_EnKF = x_ens.var(dim=0, unbiased=True)
et = time.time()
print("EnKF metrics: ", metrics_unobs(test_state_true, mu_EnKF, cov_EnKF))
print("EnKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L63_DA_EnKF.npz", mean=mu_EnKF, std=cov_EnKF**0.5)



# PF
torch.manual_seed(0)
np.random.seed(0)
particle_size = 1000
x0 = train_state[np.random.randint(0, Ntrain, size=particle_size)]
ess_frac = 0.5
st = time.time()
mu_PF, cov_PF = PF(y=test_obs, state_dim=state_dim, forecast=forecast_L63, obs_map=obs_map_L63, cov_obs=sigma_obs**2*torch.eye(obs_dim), J=particle_size, x0=x0, ess_frac=ess_frac)
et = time.time()
print("PF metrics: ", metrics_unobs(test_state_true, mu_PF, cov_PF))
print("PF time: ", et - st)
# np.savez(r"./data/VisualizationData/L63_DA_PF.npz", mean=mu_PF, std=cov_PF**0.5)
