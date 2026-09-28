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
train_u = u[:Ntrain]
test_u = u[-Ntest:]
train_u1 = u[:Ntrain, indices_u1]
test_u1 = u[-Ntest:, indices_u1]
train_u2 = u[:Ntrain, indices_u2]
test_u2 = u[-Ntest:, indices_u2]

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L96_Data.npy"), dtype=torch.float).to(device)
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]
train_u1_true = train_u_true[:, indices_u1]
test_u1_true = test_u_true[:, indices_u1]
train_u2_true = train_u_true[:, indices_u2]
test_u2_true = test_u_true[:, indices_u2]


###########################################################
#################### Data Assimilation ####################
###########################################################

test_DA_CGM = np.load(r"./data/VisualizationData/L96_DA_CGM.npz")
test_DA_mean_CGM = test_DA_CGM["mean"]
test_DA_std_CGM = test_DA_CGM["std"]
test_DA_CGKN = np.load(r"./data/VisualizationData/L96_DA_CGKN.npz")
test_DA_mean_CGKN = test_DA_CGKN["mean"]
test_DA_std_CGKN = test_DA_CGKN["std"]
test_DA_EnKF = np.load(r"./data/VisualizationData/L96_DA_EnKF.npz")
test_DA_mean_EnKF = test_DA_EnKF["mean"]
test_DA_std_EnKF = test_DA_EnKF["std"]
test_DA_PF = np.load(r"./data/VisualizationData/L96_DA_PF.npz")
test_DA_mean_PF = test_DA_PF["mean"]
test_DA_std_PF = test_DA_PF["std"]
test_DA_UKF = np.load(r"./data/VisualizationData/L96_DA_UKF.npz")
test_DA_mean_UKF = test_DA_UKF["mean"]
test_DA_std_UKF = test_DA_UKF["std"]


# DA Fig.1 Pcolor
si = 10000
ei = 10500
time_idx = np.arange(si, ei)*dt_obs
fig = plt.figure(figsize=(10, 5), constrained_layout=True)
axs = fig.subplots(3, 2)
vmin, vmax = torch.min(test_u_true[si:ei, indices_u2]).item(), torch.max(test_u_true[si:ei, indices_u2]).item()
c = axs[0,0].pcolor(time_idx, range(1, 41), test_u_true[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[0,0], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35,
cbar.ax.tick_params(labelsize=13)
vmin, vmax = test_DA_mean_CGM[si:ei, indices_u2].min(), test_DA_mean_CGM[si:ei, indices_u2].max()
c = axs[0,1].pcolor(time_idx, range(1, 41), test_DA_mean_CGM[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[0,1], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35, pad=0.03
cbar.ax.tick_params(labelsize=13)
vmin, vmax = test_DA_mean_CGKN[si:ei, indices_u2].min(), test_DA_mean_CGKN[si:ei, indices_u2].max()
axs[1,0].pcolor(time_idx, range(1, 41), test_DA_mean_CGKN[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[1,0], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35, pad=0.03
cbar.ax.tick_params(labelsize=13)
vmin, vmax = test_DA_mean_PF[si:ei, indices_u2].min(), test_DA_mean_PF[si:ei, indices_u2].max()
axs[1,1].pcolor(time_idx, range(1, 41), test_DA_mean_PF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[1,1], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35, pad=0.03
cbar.ax.tick_params(labelsize=13)
vmin, vmax = test_DA_mean_EnKF[si:ei, indices_u2].min(), test_DA_mean_EnKF[si:ei, indices_u2].max()
axs[2,0].pcolor(time_idx, range(1, 41), test_DA_mean_EnKF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[2,0], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35, pad=0.03
cbar.ax.tick_params(labelsize=13)
vmin, vmax = test_DA_mean_UKF[si:ei, indices_u2].min(), test_DA_mean_UKF[si:ei, indices_u2].max()
axs[2,1].pcolor(time_idx, range(1, 41), test_DA_mean_UKF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(c, ax=axs[2,1], pad=0.01, ticks=[-5, 0, 5, 10]) # , shrink=0.85, aspect=35, pad=0.03
cbar.ax.tick_params(labelsize=13)
[ax.axhline(y=i+1, color="cyan", linestyle="--", linewidth=0.8) for ax in axs.flat for i in indices_u1]
[ax.set_xlabel(r"$t$", fontsize=15, rotation=0) for ax in axs[-1,:]]
[ax.set_ylabel(r"$i$", fontsize=15, rotation=0, labelpad=8) for ax in axs[:,0]]
[ax.set_xticks([500, 505, 510, 515, 520, 525]) for ax in axs.flat]
[ax.set_yticks([10, 20, 30, 40]) for ax in axs.flat]
[ax.tick_params(labelsize=13, width=0.5, length=6) for ax in axs.flat]
titles = ["Truth", "Neural CGM", "CGKN", "PF", "EnKF", "UKF",]
[axs.flat[i].set_title(titles[i], fontsize=15, pad=3) for i in range(len(titles))]
# fig.savefig("./figs/L96_DA_Pcolor.png", dpi=600)


# DA Fig.2 Time Series
si = 10000
ei = 12000
idx = 15
time_idx = np.arange(si, ei)*dt_obs
fig = plt.figure(figsize=(8, 10))
axs = fig.subplots(5, 1)
axs[0].plot(time_idx, test_u_true[si:ei, idx], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[0].plot(time_idx, test_DA_mean_CGM[si:ei, idx], color="red", linewidth=1., label=r"\textbf{Neural CGM}")
axs[0].fill_between(time_idx, test_DA_mean_CGM[si:ei, idx]-2*test_DA_std_CGM[si:ei, idx], test_DA_mean_CGM[si:ei, idx]+2*test_DA_std_CGM[si:ei, idx], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[1].plot(time_idx, test_u_true[si:ei, idx], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[1].plot(time_idx, test_DA_mean_CGKN[si:ei, idx], color="red", linewidth=1., label=r"\textbf{CGKN}")
axs[1].fill_between(time_idx, test_DA_mean_CGKN[si:ei, idx]-2*test_DA_std_CGKN[si:ei, idx], test_DA_mean_CGKN[si:ei, idx]+2*test_DA_std_CGKN[si:ei, idx], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[2].plot(time_idx, test_u_true[si:ei, idx], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[2].plot(time_idx, test_DA_mean_PF[si:ei, idx], color="red", linewidth=1., label=r"\textbf{PF}")
axs[2].fill_between(time_idx, test_DA_mean_PF[si:ei, idx]-2*test_DA_std_PF[si:ei, idx], test_DA_mean_PF[si:ei, idx]+2*test_DA_std_PF[si:ei, idx], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[3].plot(time_idx, test_u_true[si:ei, idx], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[3].plot(time_idx, test_DA_mean_EnKF[si:ei, idx], color="red", linewidth=1., label=r"\textbf{EnKF}")
axs[3].fill_between(time_idx, test_DA_mean_EnKF[si:ei, idx]-2*test_DA_std_EnKF[si:ei, idx], test_DA_mean_EnKF[si:ei, idx]+2*test_DA_std_EnKF[si:ei, idx], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[4].plot(time_idx, test_u_true[si:ei, idx], color="black", linewidth=1., label=r"\textbf{Truth}")
axs[4].plot(time_idx, test_DA_mean_UKF[si:ei, idx], color="red", linewidth=1., label=r"\textbf{UKF}")
axs[4].fill_between(time_idx, test_DA_mean_UKF[si:ei, idx]-2*test_DA_std_UKF[si:ei, idx], test_DA_mean_UKF[si:ei, idx]+2*test_DA_std_UKF[si:ei, idx], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
axs[4].set_xlabel(r"$t$", fontsize=14)
methods = ["Neural CGM", "CGKN", "PF", "EnKF", "UKF"]
[axs[i].text(-0.1, 0.5, rf"\textbf{{{methods[i]}}}", transform=axs[i].transAxes, fontsize=13, ha="center", va="center", rotation=90) for i in range(len(methods))]
for ax in axs:
    ax.set_xlim([time_idx[0], time_idx[-1]+dt_obs])
    ax.set_ylim([-8., 15.])
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_ylabel(rf"$u_{{{idx + 1}}}$", fontsize=14, rotation=0, labelpad=8)
    ax.tick_params(labelsize=12, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1)
fig.tight_layout()
# fig.savefig("./figs/L96_DA_TimeSeries.png", dpi=600)


########################################################
#################### State Forecast ####################
########################################################

memory = 101
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1).numpy()
test_rollout_CGM = np.load(r"./data/VisualizationData/L96_Forecast_CGM.npy")
test_rollout_CGKN = np.load(r"./data/VisualizationData/L96_Forecast_CGKN.npy")
test_rollout_DNN = np.load(r"./data/VisualizationData/L96_Forecast_DNN.npy")


# SF Fig (Lead Time)
si = 10000
ei = 10500
idx = 15
time_idx = np.arange(si, ei)*dt_obs
fig = plt.figure(figsize=(6, 12))
axs = fig.subplots(5, 1)
lead_idx = 2
axs[0].plot(time_idx, test_rollout_true[si:ei, lead_idx, idx], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[0].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, idx], color="red", linewidth=0.8, label=r"\textbf{Neural CGM}")
axs[0].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, idx], color="blue", linewidth=0.5, label=r"\textbf{CGKN}")
axs[0].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, idx], color="green", linewidth=0.5, label=r"\textbf{DNN}")
lead_idx = 10
axs[1].plot(time_idx, test_rollout_true[si:ei, lead_idx, idx], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[1].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, idx], color="red", linewidth=0.8, label=r"\textbf{Neural CGM}")
axs[1].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, idx], color="blue", linewidth=0.5, label=r"\textbf{CGKN}")
axs[1].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, idx], color="green", linewidth=0.5, label=r"\textbf{DNN}")
lead_idx = 20
axs[2].plot(time_idx, test_rollout_true[si:ei, lead_idx, idx], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[2].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, idx], color="red", linewidth=0.8, label=r"\textbf{Neural CGM}")
axs[2].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, idx], color="blue", linewidth=0.5, label=r"\textbf{CGKN}")
axs[2].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, idx], color="green", linewidth=0.5, label=r"\textbf{DNN}")
lead_idx = 40
axs[3].plot(time_idx, test_rollout_true[si:ei, lead_idx, idx], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[3].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, idx], color="red", linewidth=0.8, label=r"\textbf{Neural CGM}")
axs[3].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, idx], color="blue", linewidth=0.5, label=r"\textbf{CGKN}")
axs[3].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, idx], color="green", linewidth=0.5, label=r"\textbf{DNN}")
lead_idx = 100
axs[4].plot(time_idx, test_rollout_true[si:ei, lead_idx, idx], color="black", linewidth=1, label=r"\textbf{Truth}")
axs[4].plot(time_idx, test_rollout_CGM[si:ei, lead_idx, idx], color="red", linewidth=0.8, label=r"\textbf{Neural CGM}")
axs[4].plot(time_idx, test_rollout_CGKN[si:ei, lead_idx, idx], color="blue", linewidth=0.5, label=r"\textbf{CGKN}")
axs[4].plot(time_idx, test_rollout_DNN[si:ei, lead_idx, idx], color="green", linewidth=0.5, label=r"\textbf{DNN}")
axs[4].set_xlabel(r"$t$", fontsize=14)
lead_time = [0.1, 0.5, 1, 2, 5]
tabs = ["(a)", "(b)", "(c)", "(d)", "(e)"]
[axs[i].set_title(rf"{tabs[i]} Forecast lead time: {lead_time[i]} time units", fontsize=12) for i in range(5)]
for ax in axs:
    ax.set_xlim([time_idx[0], time_idx[-1]+dt_obs])
    ax.set_ylim([-8., 15.])
    ax.set_yticks([-5, 0, 5, 10])
    ax.set_ylabel(rf"$u_{{{idx + 1}}}$", fontsize=14, rotation=0, labelpad=8)
    ax.tick_params(labelsize=11, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1)
fig.tight_layout()
handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, fontsize=11, loc="upper center", ncol=4, fancybox=False, edgecolor="black") #, bbox_to_anchor=(0.525, 1)
# lege.get_frame().set_linewidth(1)
fig.subplots_adjust(top=0.9) # hspace=0.3, wspace=0.22
# fig.savefig("./figs/L96_SF_TimeSeries.png", dpi=600)

