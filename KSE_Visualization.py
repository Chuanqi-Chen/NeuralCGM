# Kuramoto–Sivashinsky Equation (KSE)
# Observed States: 2 points uniformly distributed in the spatial domain with observation noise
# Unobserved States: spatial domain

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as nnF

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# mpl.use("Qt5Agg")
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

device = "cpu"
torch.manual_seed(0)
np.random.seed(0)

####################################################
################# Data Preparation #################
####################################################
# Simulation Settings: Lt=5000, Lx=22, dt=0.025, dx=22/2048
Lt = 50000
Lx = 22

# Data Resolution: dt=0.25, dx=22/2048
u = torch.tensor(np.load("data/SimulationData/KSE_Data_NoisyObs.npy"), dtype=torch.float)
t = np.arange(0, Lt, 0.25)
gridx = np.arange(0, Lx, 22/2048)


# Sub-Sampling: dt=1, dx=22/128
u = u[::4, ::16]  # (50000, 128)
t = t[::4]
gridx = gridx[::16]


# Train /Test
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]
train_t = t[:Ntrain]
test_t = t[-Ntest:]

# Observed / Unobserved
indices_u1 = np.arange(0, 128, 64) # 2 points
indices_u2 = np.arange(0, 128)[np.arange(0, 128)%64!=0] # 126 points
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
train_u1 = train_u[:, indices_u1]
test_u1 = test_u[:, indices_u1]
train_u2 = train_u[:, indices_u2]
test_u2 = test_u[:, indices_u2]

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/KSE_Data.npy"), dtype=torch.float)
u_true = u_true[::4, ::16]
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]
train_u1_true = train_u_true[:, indices_u1]
test_u1_true = test_u_true[:, indices_u1]
train_u2_true = train_u_true[:, indices_u2]
test_u2_true = test_u_true[:, indices_u2]


###########################################################
#################### Data Assimilation ####################
###########################################################

test_DA_CGM = np.load(r"./data/VisualizationData/KSE_DA_CGM.npz")
test_DA_mean_CGM = test_DA_CGM["mean"]
test_DA_std_CGM = test_DA_CGM["std"]
test_DA_CGKN = np.load(r"./data/VisualizationData/KSE_DA_CGKN.npz")
test_DA_mean_CGKN = test_DA_CGKN["mean"]
test_DA_std_CGKN = test_DA_CGKN["std"]
test_DA_EnKF = np.load(r"./data/VisualizationData/KSE_DA_EnKF.npz")
test_DA_mean_EnKF = test_DA_EnKF["mean"]
test_DA_std_EnKF = test_DA_EnKF["std"]
test_DA_PF = np.load(r"./data/VisualizationData/KSE_DA_PF.npz")
test_DA_mean_PF = test_DA_PF["mean"]
test_DA_std_PF = test_DA_PF["std"]
test_DA_EKF = np.load(r"./data/VisualizationData/KSE_DA_EKF.npz")
test_DA_mean_EKF = test_DA_EKF["mean"]
test_DA_std_EKF = test_DA_EKF["std"]
test_DA_UKF = np.load(r"./data/VisualizationData/KSE_DA_UKF.npz")
test_DA_mean_UKF = test_DA_UKF["mean"]
test_DA_std_UKF = test_DA_UKF["std"]
test_interpolation = np.load(r"./data/VisualizationData/KSE_Interpolation.npy")


# DA Fig.1 Pcolor
si = 0
ei = 500
vmin = torch.min(test_u[si:ei]).item()
vmax = torch.max(test_u[si:ei]).item()
vmax = max(abs(vmin), abs(vmax))
vmin = -vmax

fig = plt.figure(figsize=(8, 8), constrained_layout=True)
axs = fig.subplots(4, 2)
c = axs[0, 0].pcolor(range(si, ei), gridx, test_u_true[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[0,1].pcolor(range(si, ei), gridx, test_DA_mean_CGM[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,0].pcolor(range(si, ei), gridx, test_DA_mean_CGKN[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,1].pcolor(range(si, ei), gridx, test_DA_mean_PF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[2,0].pcolor(range(si, ei), gridx, test_DA_mean_EnKF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[2,1].pcolor(range(si, ei), gridx, test_DA_mean_EKF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[3,0].pcolor(range(si, ei), gridx, test_DA_mean_UKF[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[3,1].pcolor(range(si, ei), gridx, test_interpolation[si:ei].T, cmap='magma', vmin=vmin, vmax=vmax)
[ax.axhline(y=gridx[0]+0.1, color="cyan", linestyle="--", linewidth=1.5) for ax in axs.flat]
[ax.axhline(y=gridx[64], color="cyan", linestyle="--", linewidth=1.5) for ax in axs.flat]
[ax.set_xlabel(r"$t$", fontsize=14, rotation=0) for ax in axs[-1,:]]
[ax.set_ylabel(r"$x$", fontsize=14, rotation=0, labelpad=10) for ax in axs[:,0]]
[ax.set_xticks(range(si, ei+100, 100)) for ax in axs.flat]
[ax.set_yticks(range(0, 23, 4)) for ax in axs.flat]
[ax.tick_params(labelsize=12, width=0.5, length=5) for ax in axs.flat]
titles = ["Truth", "Neural CGM", "CGKN", "PF", "EnKF", "EKF", "UKF", "Interpolation"]
[axs.flat[i].set_title(titles[i], fontsize=12, pad=3) for i in range(len(titles))]
cbar = fig.colorbar(c, ax=axs, shrink=0.85, aspect=35, pad=0.03)
# fig.savefig("./figs/KSE_DA_Pcolor.png", dpi=600)



# DA Fig.2 Spatial Profile
si = 0
steps = 4
t_lst = si + steps*np.arange(6)
fig = plt.figure(figsize=(12, 8))
axs = fig.subplots(6, 6)
for n in range(6):
    axs[0, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[0, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[0, n].plot(gridx, test_DA_mean_CGM[si+steps*n], color="red", linewidth=1., label=r"\textbf{Neural CGM}")
    axs[0, n].fill_between(gridx, test_DA_mean_CGM[si+steps*n] - 2*test_DA_std_CGM[si+steps*n], test_DA_mean_CGM[si+steps*n] + 2*test_DA_std_CGM[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[0, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
    axs[1, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[1, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[1, n].plot(gridx, test_DA_mean_CGKN[si+steps*n], color="red", linewidth=1., label=r"\textbf{CGKN}")
    axs[1, n].fill_between(gridx, test_DA_mean_CGKN[si+steps*n] - 2*test_DA_std_CGKN[si+steps*n], test_DA_mean_CGKN[si+steps*n] + 2*test_DA_std_CGKN[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[1, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
    axs[2, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[2, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[2, n].plot(gridx, test_DA_mean_PF[si+steps*n], color="red", linewidth=1., label=r"\textbf{PF}")
    axs[2, n].fill_between(gridx, test_DA_mean_PF[si+steps*n] - 2*test_DA_std_PF[si+steps*n], test_DA_mean_PF[si+steps*n] + 2*test_DA_std_PF[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[2, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
    axs[3, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[3, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[3, n].plot(gridx, test_DA_mean_EnKF[si+steps*n], color="red", linewidth=1., label=r"\textbf{EnKF}")
    axs[3, n].fill_between(gridx, test_DA_mean_EnKF[si+steps*n] - 2*test_DA_std_EnKF[si+steps*n], test_DA_mean_EnKF[si+steps*n] + 2*test_DA_std_EnKF[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[3, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
    axs[4, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[4, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[4, n].plot(gridx, test_DA_mean_EKF[si+steps*n], color="red", linewidth=1., label=r"\textbf{EKF}")
    axs[4, n].fill_between(gridx, test_DA_mean_EKF[si+steps*n] - 2*test_DA_std_EKF[si+steps*n], test_DA_mean_EKF[si+steps*n] + 2*test_DA_std_EKF[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[4, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
    axs[5, n].plot(gridx, test_u_true[si+steps*n], color="black", linewidth=1., label=r"\textbf{Truth}")
    axs[5, n].scatter(gridx[indices_u1], test_u[si+steps*n, indices_u1], color="black", s=25, label=r"\textbf{Observation}")
    axs[5, n].plot(gridx, test_DA_mean_UKF[si+steps*n], color="red", linewidth=1., label=r"\textbf{UKF}")
    axs[5, n].fill_between(gridx, test_DA_mean_UKF[si+steps*n] - 2*test_DA_std_UKF[si+steps*n], test_DA_mean_UKF[si+steps*n] + 2*test_DA_std_UKF[si+steps*n], color='grey', alpha=0.5, label=r"\textbf{Uncertainty}")
    axs[5, n].plot(gridx, test_interpolation[si+steps*n], color="blue", linewidth=1., linestyle="dashed", label=r"\textbf{Interpolation}")
[ax.set_title(rf"$t={t_lst[i]}$", fontsize=20) for i, ax in enumerate(axs[0, :])]
[ax.set_xlabel(r"$x$", fontsize=18, rotation=0) for ax in axs[-1, :]]
[ax.set_ylabel(r"$u$", fontsize=18, rotation=0) for ax in axs[:, 0]]
methods = ["Neural CGM", "CGKN", "PF", "EnKF", "EKF", "UKF"]
[axs[i, 0].text(-0.38, 0.5, rf"\textbf{{{methods[i]}}}", transform=axs[i, 0].transAxes, fontsize=15, ha="center", va="center", rotation=90) for i in range(len(methods))]
for ax in axs.flat:
    ax.set_xlim([gridx[0], gridx[-1]])
    ax.set_xticks([0, 11, 22])
    ax.set_ylim([-3, 3])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=14, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1)
fig.tight_layout()
# fig.savefig("./figs/KSE_DA_SpatialProfile.png", dpi=600)




########################################################
#################### State Forecast ####################
########################################################

memory = 21
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1)
test_rollout_CGM = np.load(r"./data/VisualizationData/KSE_Forecast_CGM.npy")[:-memory+1]
test_rollout_CGKN = np.load(r"./data/VisualizationData/KSE_Forecast_CGKN.npy")[:-memory+1]
test_rollout_FNO = np.load(r"./data/VisualizationData/KSE_Forecast_FNO.npy")[:-memory+1]
test_rollout_CNN = np.load(r"./data/VisualizationData/KSE_Forecast_CNN.npy")[:-memory+1]

# SF Fig
si = 0
step = 1
idx = 8000
fig = plt.figure(figsize=(12, 6))
axs = fig.subplots(3, 5)
t_lst = si + step*np.arange(axs.size)
for n in range(axs.size):
    axs.flat[n].plot(gridx, test_rollout_true[idx, si + n*step], color="black", linewidth=2.5, linestyle="dashed", label=r"\textbf{Truth}")
    if n > 0:
        axs.flat[n].plot(gridx, test_rollout_CNN[idx, si + n * step], color="darkviolet", linewidth=1.5, label=r"\textbf{CNN}")
        axs.flat[n].plot(gridx, test_rollout_CGKN[idx, si + n*step], color="blue", linewidth=1.5, label=r"\textbf{CGKN}")
        axs.flat[n].plot(gridx, test_rollout_FNO[idx, si + n*step], color="limegreen", linewidth=1.5, label=r"\textbf{FNO}")
        axs.flat[n].plot(gridx, test_rollout_CGM[idx, si + n*step], color="red", linewidth=1.5, label=r"\textbf{Neural CGM}")
    axs.flat[n].set_title(rf"$t={t_lst[n]}$", fontsize=18)
    axs.flat[n].set_xlim([gridx[0], gridx[-1]])
    axs.flat[n].set_xticks([0, 11, 22])
    axs.flat[n].set_ylim([-3.3, 3.3])
    axs.flat[n].set_yticks([-2, 0, 2])
    axs.flat[n].tick_params(labelsize=15, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    # for spine in axs.flat[n].spines.values():
    #     spine.set_linewidth(1)
[ax.set_xlabel(r"$x$", fontsize=18) for ax in axs[-1, :]]
[ax.set_ylabel(r"$u$", fontsize=18, rotation=0) for ax in axs[:, 0]]
fig.tight_layout()
handles, labels = axs[0, 1].get_legend_handles_labels()
handles = [handles[0], handles[4], handles[2], handles[1], handles[3]]
labels = [labels[0], labels[4], labels[2], labels[1], labels[3]]
lege = fig.legend(handles, labels, fontsize=16, loc="upper center", ncol=5, fancybox=False, edgecolor="black") #, bbox_to_anchor=(0.525, 1)
# lege.get_frame().set_linewidth(1)
fig.subplots_adjust(top=0.85) # hspace=0.3, wspace=0.22
# fig.savefig("./figs/KSE_SF_SpatialProfile.png", dpi=600)




##############################################################
################# NeuralCGM VS. CGKN (dim_v) #################
##############################################################

import re
text = open(r"./data/VisualizationData/NeuralCGM_VS_CGKN").read()

def extract(model, metric):
    return np.array([ float(x) for x in re.findall( rf"{model}:.*?{metric}:\s+([\d.eE+-]+)" rf"(?=.*?(?:dim_v\s*=|\Z))", text, re.S ) ])

dim_v_list = np.arange(4, 16)
test_forecast_RMSE_CGM = extract("NeuralCGM", "test forecast RMSE")
test_DA_RMSE_CGM       = extract("NeuralCGM", "test DA RMSE")
test_DA_NLL_CGM        = extract("NeuralCGM", "test DA NLL")
param_CGM              = extract("NeuralCGM", "params count").astype(int)
test_forecast_RMSE_CGKN = extract("CGKN", "test forecast RMSE")
test_DA_RMSE_CGKN       = extract("CGKN", "test DA RMSE")
test_DA_NLL_CGKN        = extract("CGKN", "test DA NLL")
param_CGKN              = extract("CGKN", "params count").astype(int)

fig = plt.figure(figsize=(8, 6))
axs = fig.subplots(2, 2)
axs[0, 0].plot(dim_v_list, test_forecast_RMSE_CGM, marker="o", markersize=5, color="red", label="Neural CGM")
axs[0, 0].plot(dim_v_list, test_forecast_RMSE_CGKN, marker="s", markersize=5, color="blue", label="CGKN")
axs[0, 1].plot(dim_v_list, test_DA_RMSE_CGM, marker="o", markersize=5, color="red", label="Neural CGM")
axs[0, 1].plot(dim_v_list, test_DA_RMSE_CGKN, marker="s", markersize=5, color="blue", label="CGKN")
axs[1, 0].plot(dim_v_list, test_DA_NLL_CGM, marker="o", markersize=5, color="red", label="Neural CGM")
axs[1, 0].plot(dim_v_list, test_DA_NLL_CGKN, marker="s", markersize=5, color="blue", label="CGKN")
axs[1, 1].plot(dim_v_list, param_CGM, marker="o", markersize=5, color="red", label="Neural CGM")
axs[1, 1].plot(dim_v_list, param_CGKN, marker="s", markersize=5, color="blue", label="CGKN")
[ax.set_xlabel("Dimension of $\mathbf{z}$", fontsize=14) for ax in axs.flat]
[ax.set_xticks(range(4, 16)) for ax in axs.flat]
axs[0, 0].set_ylabel(r"Forecast RMSE", fontsize=14)
axs[0, 1].set_ylabel(r"DA RMSE", fontsize=14)
axs[1, 0].set_ylabel(r"DA NLL", fontsize=14)
axs[1, 1].set_ylabel(r"NO. of parameters", fontsize=14)
for ax in axs.flat:
    ax.tick_params(labelsize=14, length=5, width=0.5, direction="in", top=True, bottom=True, left=True, right=True)
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
axs[0, 0].legend(fontsize=15)
fig.tight_layout()
yticks = [ax.get_yticks()[1:-1] for ax in axs.flat]
fig.subplots_adjust(top=0.935)
fig.subplots_adjust(hspace=0.35)
[ax.text(-0.1, 1.15, f"({label})", transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="top") for ax, label in zip(axs.flat, ["a", "b", "c", "d"])]
[ax.set_yticks(ticks) for ax, ticks in zip(axs.flat, yticks)]
axs[1, 1].set_yticks(1000*np.arange(95, 125, 5))
axs[1, 1].set_ylim([95*1000, 121*1000])
axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(3, 3), useMathText=True)
axs[0, 0].set_yticks(np.arange(0.05, 0.18, 0.02))
axs[0, 0].set_ylim([0.04, 0.17])
# axs[0, 1].set_yticks(np.arange(20, 220, 20))
axs[0, 1].set_ylim([0.35, 0.75])
axs[1, 0].set_yticks(np.arange(20, 220, 20))
axs[1, 0].set_ylim([30, 190])
# fig.savefig("./figs/KSE_dim_z.png", dpi=600)
