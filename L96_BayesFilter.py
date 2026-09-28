# Stochastic Lorenz 96 System (40 States) with Noisy Observation
# Observed States: 4 states uniformly distributed in state space
# Unobserved States: other 36 states

import numpy as np
import scipy as sp
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
torch.manual_seed(0)
np.random.seed(0)

####################################################
################# Data Preparation #################
####################################################

u = torch.tensor(np.load("data/SimulationData/L96_Data_NoisyObs.npy"), dtype=torch.float).to(device)
sigma_obs = 0.75
sigma_dyn = 0.5
dt_obs = 0.05

# Train/Test & State/Observation
Nt_obs, state_size = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
indices_u1 = np.arange(0, 40, 10) # 4 states
indices_u2 = np.arange(0, 40)[np.arange(0, 40)%10!=0] # 36 states
obs_size = len(indices_u1)
train_x = u[:Ntrain]
test_x = u[-Ntest:]
train_y = u[:Ntrain, indices_u1]
test_y = u[-Ntest:, indices_u1]


##########################################################
################# State-Space Model ######################
##########################################################

def dynamics_L96(u):
    # u shape (N, state_dim)
    F = 8.
    u_im1 = torch.roll(u, shifts=1, dims=1)    # u[:, i-1]
    u_ip1 = torch.roll(u, shifts=-1, dims=1)   # u[:, i+1]
    u_im2 = torch.roll(u, shifts=2, dims=1)    # u[:, i-2]
    u_dot = -u + u_ip1 * u_im1 - u_im2 * u_im1 + F
    return u_dot

def obs_map_L96(u):
    # u: (N, S)
    return u[:, indices_u1]

def forecast_L96(u, delta_t=dt_obs):
    # u shape (N, S)
    F = 8
    sigma_dyn = 0.5
    dt_sim = 0.001
    Nt_sim = int(delta_t / dt_sim)
    for _ in range(Nt_sim):
        u_im1 = torch.roll(u, shifts=1, dims=1)    # u[:, i-1]
        u_ip1 = torch.roll(u, shifts=-1, dims=1)   # u[:, i+1]
        u_im2 = torch.roll(u, shifts=2, dims=1)    # u[:, i-2]
        u_dot = -u + u_ip1 * u_im1 - u_im2 * u_im1 + F
        u = u + u_dot * dt_sim + sigma_dyn * dt_sim**0.5 * torch.randn_like(u)
    return u

def Jacobian_dyn_L96(u):
    # u: (B, state_dim)
    B, state_dim = u.shape

    J = torch.zeros(B, state_dim, state_dim, device=u.device, dtype=u.dtype)

    idx = torch.arange(state_dim, device=u.device)

    im1 = (idx - 1) % state_dim
    ip1 = (idx + 1) % state_dim
    im2 = (idx - 2) % state_dim

    # d f_i / d x_i
    J[:, idx, idx] = -1.0

    # d f_i / d x_{i+1}
    J[:, idx, ip1] = u[:, im1]

    # d f_i / d x_{i-2}
    J[:, idx, im2] = -u[:, im1]

    # d f_i / d x_{i-1}
    J[:, idx, im1] = u[:, ip1] - u[:, im2]

    return J

def Jacobian_obs_L96(u):
    # u: (B, state_dim)
    B, state_dim = u.shape
    obs_dim = len(indices_u1)
    H = torch.zeros(B, obs_dim, state_dim, device=u.device, dtype=u.dtype)
    obs_idx = torch.tensor(indices_u1, device=u.device, dtype=torch.long)
    H[:, torch.arange(obs_dim, device=u.device), obs_idx] = 1.0
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
u_true = torch.tensor(np.load("data/SimulationData/L96_Data.npy"), dtype=torch.float).to(device)
train_x_true = u_true[:Ntrain]
test_x_true = u_true[-Ntest:]
train_y_true = u_true[:Ntrain, indices_u1]
test_y_true = u_true[-Ntest:, indices_u1]


# EKF
torch.manual_seed(0)
np.random.seed(0)
mu0 = train_x.mean(dim=0)
cov0 = torch.cov(train_x.T)
dt_sim = 0.001
st = time.time()
mu_EKF, cov_EKF = EKF(y=test_y, state_dim=state_size, forecast_dyn=dynamics_L96, obs_map=obs_map_L96, cov_dyn=sigma_dyn**2 * torch.eye(state_size), cov_obs=sigma_obs ** 2 * torch.eye(obs_size), dt_obs=dt_obs, dt_sim=dt_sim, Jacobian_dyn=Jacobian_dyn_L96, Jacobian_obs=Jacobian_obs_L96, mu0=mu0, cov0=cov0)
et = time.time()
print("EKF metrics: ", metrics_unobs(test_x_true, mu_EKF, cov_EKF))
print("EKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L96_DA_EKF.npz", mean=mu_EKF, std=cov_EKF.diagonal(dim1=-2, dim2=-1)**0.5)


# UKF
torch.manual_seed(0)
np.random.seed(0)
mu0 = train_x.mean(dim=0)
cov0 = torch.cov(train_x.T)
dt_sim = 0.001
st = time.time()
mu_UKF, cov_UKF = UKF(y=test_y, state_dim=state_size, forecast_dyn=dynamics_L96, obs_map=obs_map_L96, cov_dyn=sigma_dyn**2*torch.eye(state_size), cov_obs=sigma_obs ** 2 * torch.eye(obs_size), dt_obs=dt_obs, dt_sim=dt_sim, alpha=0.72, k=0, beta=2, mu0=mu0, cov0=cov0)
et = time.time()
print("UKF metrics: ", metrics_unobs(test_x_true, mu_UKF, cov_UKF))
print("UKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L96_DA_UKF.npz", mean=mu_UKF, std=cov_UKF.diagonal(dim1=-2, dim2=-1)**0.5)



# EnKF
torch.manual_seed(0)
np.random.seed(0)
ens_size = 200
obs_inflation = 2.
x0 = train_x[np.random.randint(0, Ntrain, size=ens_size)]
st = time.time()
x_ens = EnKF(y=test_y, state_dim=state_size, forecast=forecast_L96, obs_map=obs_map_L96, cov_obs=(obs_inflation*sigma_obs)**2*torch.eye(obs_size).to(device), J=ens_size, x0=x0, inflation=None, process_noise_coef=None)
mu_EnKF = x_ens.mean(dim=0)
# cov_EnKF = torch.einsum("nti,ntj->tij", x_ens-mu_EnKF, x_ens-mu_EnKF) / (x_ens.shape[0]-1)
cov_EnKF = x_ens.var(dim=0, unbiased=True)
et = time.time()
print("EnKF metrics: ", metrics_unobs(test_x_true, mu_EnKF, cov_EnKF))
print("EnKF time: ", et - st)
# np.savez(r"./data/VisualizationData/L96_DA_EnKF.npz", mean=mu_EnKF, std=cov_EnKF**0.5)



# PF
torch.manual_seed(0)
np.random.seed(0)
particle_size = 1000
ess_frac = 0.7
likelihood_inflation = 20.
roughening = 0.01
x0 = train_x[np.random.randint(0, Ntrain, size=particle_size)]
st = time.time()
mu_PF, cov_PF = PF(y=test_y, state_dim=state_size, forecast=forecast_L96, obs_map=obs_map_L96, cov_obs=(likelihood_inflation*sigma_obs)**2*torch.eye(obs_size), J=particle_size, x0=x0, ess_frac=ess_frac, roughening=roughening)
et = time.time()
print("PF metrics: ", metrics_unobs(test_x_true, mu_PF, cov_PF))
print("PF time: ", et - st)
# np.savez(r"./data/VisualizationData/L96_DA_PF.npz", mean=mu_PF, std=cov_PF**0.5)

