# Lorenz 63 System
# Observed States: x
# Unobserved States: [y, z]

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

u = torch.tensor(np.load("data/SimulationData/L63_Data_NoisyObs.npy"), dtype=torch.float)
dt_obs = 0.05

# Train /Test
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]

# Observed / UnObserved
indices_u1 = [0]
indices_u2 = [1, 2]
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
train_u1 = train_u[:, indices_u1]
test_u1 = test_u[:, indices_u1]
train_u2 = train_u[:, indices_u2]
test_u2 = test_u[:, indices_u2]

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L63_Data.npy"), dtype=torch.float)
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]
train_u1_true = train_u_true[:, indices_u1]
test_u1_true = test_u_true[:, indices_u1]
train_u2_true = train_u_true[:, indices_u2]
test_u2_true = test_u_true[:, indices_u2]



###########################################################
#################### Data Assimilation ####################
###########################################################

test_DA_CGM = np.load(r"./data/VisualizationData/L63_DA_CGM.npz")
test_DA_mean_CGM = test_DA_CGM["mean"]
test_DA_std_CGM = test_DA_CGM["std"]
test_DA_CGKN = np.load(r"./data/VisualizationData/L63_DA_CGKN.npz")
test_DA_mean_CGKN = test_DA_CGKN["mean"]
test_DA_std_CGKN = test_DA_CGKN["std"]
test_DA_EnKF = np.load(r"./data/VisualizationData/L63_DA_EnKF.npz")
test_DA_mean_EnKF = test_DA_EnKF["mean"]
test_DA_std_EnKF = test_DA_EnKF["std"]
test_DA_PF = np.load(r"./data/VisualizationData/L63_DA_PF.npz")
test_DA_mean_PF = test_DA_PF["mean"]
test_DA_std_PF = test_DA_PF["std"]
test_DA_EKF = np.load(r"./data/VisualizationData/L63_DA_EKF.npz")
test_DA_mean_EKF = test_DA_EKF["mean"]
test_DA_std_EKF = test_DA_EKF["std"]
test_DA_UKF = np.load(r"./data/VisualizationData/L63_DA_UKF.npz")
test_DA_mean_UKF = test_DA_UKF["mean"]
test_DA_std_UKF = test_DA_UKF["std"]


# DA Fig. Time Series
si = 1000
ei = 1200
time_idx = np.arange(si, ei)*dt_obs
fig = plt.figure(figsize=(8, 4))
axs = fig.subplots(3, 2)
axs[0, 0].plot(time_idx, test_u[si:ei, 0], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[1, 0].plot(time_idx, test_u_true[si:ei, 1], color="black", linewidth=1.2, label=r"\textbf{Truth}")
axs[1, 0].plot(time_idx, test_DA_mean_CGM[si:ei, 1], color="red", linewidth=0.8, linestyle="dashed", label=r"\textbf{NeuralCGM}")
axs[1, 0].fill_between(time_idx, test_DA_mean_CGM[si:ei, 1]-2*test_DA_std_CGM[si:ei, 1], test_DA_mean_CGM[si:ei, 1]+2*test_DA_std_CGM[si:ei, 1], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[2, 0].plot(time_idx, test_u_true[si:ei, 2], color="black", linewidth=1.2, label=r"\textbf{Truth}")
axs[2, 0].plot(time_idx, test_DA_mean_CGM[si:ei, 2], color="red", linewidth=0.8, linestyle="dashed", label=r"\textbf{NeuralCGM}")
axs[2, 0].fill_between(time_idx, test_DA_mean_CGM[si:ei, 2]-2*test_DA_std_CGM[si:ei, 2], test_DA_mean_CGM[si:ei, 2]+2*test_DA_std_CGM[si:ei, 2], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[0, 1].plot(time_idx, test_u[si:ei, 0], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[1, 1].plot(time_idx, test_u_true[si:ei, 1], color="black", linewidth=1.2, label=r"\textbf{Truth}")
axs[1, 1].plot(time_idx, test_DA_mean_PF[si:ei, 1], color="red", linewidth=0.8, linestyle="dashed", label=r"\textbf{NeuralPF}")
axs[1, 1].fill_between(time_idx, test_DA_mean_PF[si:ei, 1]-2*test_DA_std_PF[si:ei, 1], test_DA_mean_PF[si:ei, 1]+2*test_DA_std_PF[si:ei, 1], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[2, 1].plot(time_idx, test_u_true[si:ei, 2], color="black", linewidth=1.2, label=r"\textbf{Truth}")
axs[2, 1].plot(time_idx, test_DA_mean_PF[si:ei, 2], color="red", linewidth=0.8, linestyle="dashed", label=r"\textbf{NeuralPF}")
axs[2, 1].fill_between(time_idx, test_DA_mean_PF[si:ei, 2]-2*test_DA_std_PF[si:ei, 2], test_DA_mean_PF[si:ei, 2]+2*test_DA_std_PF[si:ei, 2], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[2, 0].set_xlabel(r"$t$", fontsize=14)
axs[2, 1].set_xlabel(r"$t$", fontsize=14)
[ax.set_ylabel(name, fontsize=14, rotation=0) for name, ax in zip([r"$x$", r"$y$", r"$z$"], axs[:,0])]
axs[2, 0].set_ylabel(r"$z$", fontsize=14, rotation=0, labelpad=12)
axs[0, 0].set_title("Neural CGM", fontsize=13)
axs[0, 1].set_title("Particle Filter", fontsize=13)
for ax in axs.flat:
    ax.set_xlim([time_idx[0], time_idx[-1]+dt_obs])
    ax.tick_params(labelsize=10, length=4, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1)
[ax.set_yticks([-20, 0, 20]) for ax in axs[1, :]]
[ax.set_yticks([0, 20, 40]) for ax in axs[2, :]]
fig.tight_layout()
# fig.savefig("./figs/L63_DA_TimeSeries.png", dpi=600)





########################################################
#################### State Forecast ####################
########################################################

memory = 101
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1).numpy()
test_rollout_CGM = np.load(r"./data/VisualizationData/L63_Forecast_CGM.npy")
test_rollout_CGKN = np.load(r"./data/VisualizationData/L63_Forecast_CGKN.npy")
test_rollout_DNN = np.load(r"./data/VisualizationData/L63_Forecast_DNN.npy")

# SF Fig (Lead Time)
si = 1000
ei = 1600
time_idx = np.arange(si, ei)*dt_obs
fig = plt.figure(figsize=(6, 4))
axs = fig.subplots(3, 1)
lead_idx = 4
axs[0].plot(time_idx, test_rollout_true[si:ei, lead_idx, 0], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[0].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, 0], color="red", linewidth=1, label=r"\textbf{Neural CGM}")
# axs[0].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, 0], color="blue", linewidth=1, label=r"\textbf{CGKN}")
axs[0].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, 0], color="green", linewidth=0.6, label=r"\textbf{DNN}")
axs[1].plot(time_idx, test_rollout_true[si:ei, lead_idx, 1], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[1].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, 1], color="red", linewidth=1, label=r"\textbf{Neural CGM}")
# axs[1].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, 1], color="blue", linewidth=1, label=r"\textbf{CGKN}")
axs[1].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, 1], color="green", linewidth=0.6, label=r"\textbf{DNN}")
axs[2].plot(time_idx, test_rollout_true[si:ei, lead_idx, 2], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[2].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, 2], color="red", linewidth=1, label=r"\textbf{NeuralCGM}")
# axs[2].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, 2], color="blue", linewidth=1, label=r"\textbf{CGKN}")
axs[2].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, 2], color="green", linewidth=0.6, label=r"\textbf{DNN}")
axs[0].set_title("Forecast at a lead time of 0.2 time units", fontsize=12, pad=30)
[ax.set_ylabel(name, fontsize=11, rotation=0) for name, ax in zip([r"$x$", r"$y$", r"$z$"], axs)]
axs[2].set_ylabel(r"$z$", fontsize=11, rotation=0, labelpad=12)
axs[2].set_xlabel(r"$t$", fontsize=11)
for ax in axs:
    ax.set_xlim([time_idx[0], time_idx[-1]+dt_obs])
    ax.tick_params(labelsize=9, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1)
handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, fontsize=9, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.525, 0.915))
lege.get_frame().set_linewidth(0.5)
axs[1].set_yticks([-20, 0, 20])
axs[2].set_yticks([0, 20, 40])
fig.tight_layout()
# fig.savefig("./figs/L63_SF_TimeSeries.png", dpi=600)

