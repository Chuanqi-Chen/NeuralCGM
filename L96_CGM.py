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

# 4 observed states ----> full 40 states
u = torch.tensor(np.load("data/SimulationData/L96_Data_NoisyObs.npy"), dtype=torch.float).to(device)
indices_obs = np.arange(0, 40, 10)
u = torch.cat([u[:, indices_obs], u], dim=-1)  # Augmented States (dim=44)


# Train / Test
Nt_obs, state_dim = u.shape
Ntrain = int(Nt_obs*0.8)
Ntest = int(Nt_obs*0.2)
train_u = u[:Ntrain]
test_u = u[-Ntest:]
indices_u1 = np.arange(4)
indices_u2 = np.arange(4, 44)
indices_u2_unobs = np.arange(0, 40)[np.arange(0, 40)%10!=0] # 36 states
dim_u1 = len(indices_u1)
dim_u2 = len(indices_u2)
train_u1 = train_u[:, indices_u1]
test_u1 = test_u[:, indices_u1]
train_u2 = train_u[:, indices_u2]
test_u2 = test_u[:, indices_u2]

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
train_u1_normalized = train_u_normalized[:, indices_u1]
test_u1_normalized = test_u_normalized[:, indices_u1]
train_u2_normalized = train_u_normalized[:, indices_u2]
test_u2_normalized = test_u_normalized[:, indices_u2]


####################################################
################# Model Architecture ###############
####################################################

class AutoEncoder(nn.Module):
    def __init__(self, dim_u2, dim_v):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim_u2, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, dim_v))
        self.decoder = nn.Sequential(nn.Linear(dim_v, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, dim_u2))

class CGNet(nn.Module):
    def __init__(self, dim_u1, dim_v):
        super().__init__()

        self.dim_u1 = dim_u1
        self.dim_v = dim_v

        # Constant learnable parameters, independent of u1
        self.f1 = nn.Parameter(1 / dim_u1**0.5 * torch.rand(dim_u1))
        self.g1 = nn.Parameter(1 / (dim_u1 * dim_v)**0.5 * torch.rand(dim_u1, dim_v))
        self.f2 = nn.Parameter(1 / dim_v**0.5 * torch.rand(dim_v))
        self.g2 = nn.Parameter(1 / dim_v * torch.rand(dim_v, dim_v))

    def forward(self, u1):
        # u1 is only used to get batch size
        # u1: (N, dim_u1)
        batch_size = u1.shape[0]
        f1 = self.f1.unsqueeze(0).expand(batch_size, -1)
        g1 = self.g1.unsqueeze(0).expand(batch_size, -1, -1)
        f2 = self.f2.unsqueeze(0).expand(batch_size, -1)
        g2 = self.g2.unsqueeze(0).expand(batch_size, -1, -1)
        return [f1, g1, f2, g2]

class GaussianDecoder(nn.Module):
    def __init__(self, dim_u1, dim_u2, dim_v):
        super().__init__()
        # input_size = dim_v + int(dim_v*(dim_v+1)/2)
        input_size = dim_v + dim_v
        self.mean_decoder = nn.Sequential(nn.Linear(input_size+dim_u1, 16), nn.SiLU(),
                                          nn.Linear(16, 64), nn.SiLU(),
                                          nn.Linear(64, 64), nn.SiLU(),
                                          nn.Linear(64, 64), nn.SiLU(),
                                          nn.Linear(64, 64), nn.SiLU(),
                                          nn.Linear(64, dim_u2))
        self.cov_decoder = nn.Sequential(nn.Linear(input_size+dim_u1, 16), nn.SiLU(),
                                         nn.Linear(16, 64), nn.SiLU(),
                                         nn.Linear(64, 64), nn.SiLU(),
                                         nn.Linear(64, 64), nn.SiLU(),
                                         nn.Linear(64, 64), nn.SiLU(),
                                         nn.Linear(64, dim_u2))

    def decode_mean(self, mu_DA_v, cov_DA_v, u1):
        # mu_DA_V shape (N, dim_v)
        # cov_DA_v shape (N, dim_v, dim_v)
        # u1 shape (N, dim_u1)
        dim_v = mu_DA_v.shape[1]
        # idx = torch.triu_indices(dim_v, dim_v, device=cov_DA_v.device)
        # Gaussian_stats = torch.cat([mu_DA_v, cov_DA_v[:, idx[0], idx[1]], u1], dim=-1)
        Gaussian_stats = torch.cat([mu_DA_v, cov_DA_v.diagonal(dim1=1, dim2=2), u1], dim=-1)
        return self.mean_decoder(Gaussian_stats)

    def decode_cov(self, mu_DA_v, cov_DA_v, u1):
        # mu_DA_v shape (N, dim_v)
        # cov_DA_v shape (N, dim_v, dim_v)
        # u1 shape (N, dim_u1)
        dim_v = mu_DA_v.shape[1]
        # idx = torch.triu_indices(dim_v, dim_v, device=cov_DA_v.device)
        # Gaussian_stats = torch.cat([mu_DA_v, cov_DA_v[:, idx[0], idx[1]], u1], dim=-1)
        Gaussian_stats = torch.cat([mu_DA_v, cov_DA_v.diagonal(dim1=1, dim2=2), u1], dim=-1)
        return self.cov_decoder(Gaussian_stats)

class CGModel(nn.Module):
    def __init__(self, autoencoder, cgn, Gdecoder=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.cgn = cgn
        self.sigma_u1 = None # diag matrix in vector form
        self.sigma_v = None # diag matrix in vector form
        self.sigma_u2 = None # diag matrix in vector form
        self.Gdecoder = Gdecoder

    def forecast(self, uCG_init):
        # uCG_init = [u1_int, v_init]: (N, dim_u1+dim_v)
        dim_u1 = self.cgn.dim_u1
        u1_init = uCG_init[:, :dim_u1]
        v_init = uCG_init[:, dim_u1:]
        f1, g1, f2, g2 = self.cgn(u1_init)
        v_init = v_init.unsqueeze(-1) # dimension match for matrix multiplication
        u1_next = f1 + (g1@v_init).squeeze(-1)
        v_next = f2 + (g2@v_init).squeeze(-1)
        uCG_next = torch.cat([u1_next, v_next], dim=-1)
        return uCG_next

    def filter(self, u1, mu0=None, cov0=None):
        # u1: (N, T, dim_u1)
        # mu0: (N, dim_v)
        # cov0: (N, dim_v, dim_v)
        device = u1.device
        sigma_u1 = torch.diag_embed(self.sigma_u1).to(device)
        sigma_v = torch.diag_embed(self.sigma_v).to(device)
        N, Nt, dim_u1 = u1.shape
        dim_v = self.cgn.dim_v
        if mu0 is None:
            mu0 = torch.zeros(N, dim_v, device=device)
        else:
            mu0 = mu0.to(device=device)
        if cov0 is None:
            cov0 = torch.eye(dim_v, device=device)[None].repeat(N, 1, 1)
        else:
            cov0 = cov0.to(device=device)
        mu_post = torch.zeros((N, Nt, dim_v), device=device)
        cov_post = torch.zeros((N, Nt, dim_v, dim_v), device=device)
        mu_post[:, 0] = mu0
        cov_post[:, 0] = cov0
        for nt in range(1, Nt):
            mu0 = mu0.unsqueeze(-1)
            f1, g1, f2, g2 = self.cgn(u1[:, nt-1])
            f1 = f1.unsqueeze(-1)
            f2 = f2.unsqueeze(-1)
            K = g2 @ cov0 @ g1.mT @ torch.inverse(sigma_u1 @ sigma_u1.T + g1 @ cov0 @ g1.mT)
            mu1 = f2 + g2 @ mu0 +  K @ (u1[:, nt, :, None] - f1 - g1 @ mu0)
            cov1 = g2 @ cov0 @ g2.mT + sigma_v @ sigma_v.T - K @ g1 @ cov0 @ g2.mT
            mu1 = mu1.squeeze(-1)
            mu_post[:, nt] = mu1
            cov_post[:, nt] = cov1
            mu0 = mu1
            cov0 = cov1
        return (mu_post, cov_post)


    def decode_mean(self, mu_DA_V, cov_DA_v, u1):
        # mu_DA_V: (N, dim_v)
        # mu_cov: (N, dim_v, dim_v)
        # u1: (N, dim_u1)
        mu_DA_u2 = self.autoencoder.decoder(mu_DA_V) + self.Gdecoder.decode_mean(mu_DA_V, cov_DA_v, u1)
        return mu_DA_u2

    def decode_cov(self, mu_DA_v, cov_DA_v, u1):
        # mu_DA_v:  (N, dim_v)
        # cov_DA_v: (N, dim_v, dim_v)
        # u1: (N, dim_u1)
        J = torch.func.vmap(torch.func.jacrev(self.autoencoder.decoder))(mu_DA_v)
        cov_DA_u2 = torch.einsum("bod,bde,boe->bo", J, cov_DA_v, J) # diag(J @ cov_DA_v @ J.T)
        cov_DA_u2 = cov_DA_u2 + self.Gdecoder.decode_cov(mu_DA_v, cov_DA_v, u1)
        cov_DA_u2 = nnF.softplus(cov_DA_u2)
        return cov_DA_u2


################################################
################# Model Training ###############
################################################

dim_v = 80

autoencoder = AutoEncoder(dim_u2, dim_v).to(device)
cgn = CGNet(dim_u1, dim_v).to(device)
Gdecoder = GaussianDecoder(dim_u1, dim_u2, dim_v).to(device)
cg_model = CGModel(autoencoder, cgn, Gdecoder).to(device)


# Training Stage 0: Train CGModel with loss_ae
epochs = 2000
batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_u_normalized)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history_ae = []

optimizer = torch.optim.Adam(cg_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()
    train_loss_ae = 0.
    for u, in train_loader:
        u = u.to(device)
        # AutoEncoder
        u2 = u[:, indices_u2]
        v = cg_model.autoencoder.encoder(u2)
        u2_ae = cg_model.autoencoder.decoder(v)
        loss_ae = nnF.mse_loss(u2, u2_ae)
        loss_ae.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_ae += loss_ae.item()
    train_loss_ae /= train_num_batches
    train_loss_history_ae.append(train_loss_ae)

    end_time = time.time()
    print(f"ep {ep} | "
          f"time: {end_time - start_time:.4f} | "
          f"loss AE: {train_loss_ae:.4f} | ")
"""

# Training Stage 1: Train CGModel with loss_ae + loss_forecast_u + loss_forecast_z
epochs = 2000
batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_u_normalized[:-1], train_u_normalized[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history_ae = []
train_loss_history_forecast_u = []
train_loss_history_forecast_v = []

optimizer = torch.optim.Adam(cg_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_ae = 0.
    train_loss_forecast_u = 0.
    train_loss_forecast_v = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)

        # AutoEncoder
        u2 = u[:, indices_u2]
        v = cg_model.autoencoder.encoder(u2)
        u2_ae = cg_model.autoencoder.decoder(v)
        loss_ae = nnF.mse_loss(u2, u2_ae)

        #  State Forecast
        u1 = u[:, indices_u1]
        uCG = torch.cat([u1, v], dim=-1)
        uCG_pred = cg_model.forecast(uCG)
        u1_pred = uCG_pred[:, :dim_u1]
        v_pred = uCG_pred[:, dim_u1:]
        u2_pred = cg_model.autoencoder.decoder(v_pred)
        u_pred = torch.zeros_like(u)
        u_pred[:, indices_u1] = u1_pred
        u_pred[:, indices_u2] = u2_pred
        loss_forecast_u = nnF.mse_loss(u_next, u_pred)

        u2_next = u_next[:, indices_u2]
        v_next = cg_model.autoencoder.encoder(u2_next)
        loss_forecast_v = nnF.mse_loss(v_next, v_pred)

        loss_total = loss_ae + loss_forecast_u + loss_forecast_v
        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_ae += loss_ae.item()
        train_loss_forecast_u += loss_forecast_u.item()
        train_loss_forecast_v += loss_forecast_v.item()
    train_loss_ae /= train_num_batches
    train_loss_forecast_u /= train_num_batches
    train_loss_forecast_v /= train_num_batches
    train_loss_history_ae.append(train_loss_ae)
    train_loss_history_forecast_u.append(train_loss_forecast_u)
    train_loss_history_forecast_v.append(train_loss_forecast_v)

    end_time = time.time()
    print(f"ep {ep} | "
          f"time: {end_time - start_time:.4f} | "
          f"loss AE: {train_loss_ae:.4f} | "
          f"loss u: {train_loss_forecast_u:.4f} | "
          f"loss v: {train_loss_forecast_v:.4f}")
"""

# torch.save(cg_model.state_dict(), r"./model/L96_CGM_v80_stage1.pt")
# np.save(r"./model/L96_CGM_v80_train_loss_history_AE_stage1.npy", train_loss_history_ae)
# np.save(r"./model/L96_CGM_v80_train_loss_history_forecast_u_stage1.npy", train_loss_history_forecast_u)
# np.save(r"./model/L96_CGM_v80_train_loss_history_forecast_v_stage1.npy", train_loss_history_forecast_v)

cg_model.load_state_dict(torch.load(r"./model/L96_CGM_v80_stage1.pt"))

with torch.no_grad():
    train_v = cg_model.autoencoder.encoder(train_u2_normalized)
    train_uCG = torch.cat([train_u1_normalized, train_v], dim=-1)
    train_uCG_pred = cg_model.forecast(train_uCG)
    train_u1_normalized_pred = train_uCG_pred[:, :dim_u1]
    train_v_pred = train_uCG_pred[:, dim_u1:]
    train_u2_normalized_pred = cg_model.autoencoder.decoder(train_v_pred)
    train_u_normalized_pred = torch.zeros_like(train_u_normalized)
    train_u_normalized_pred[:, indices_u1] = train_u1_normalized_pred
    train_u_normalized_pred[:, indices_u2] = train_u2_normalized_pred
    train_u_pred = Normalizer.decode(train_u_normalized_pred)
nnF.mse_loss(train_u_pred[:-1], train_u[1:]).item()

cg_model.sigma_u1 = torch.sqrt(torch.mean((train_u1_normalized[1:] - train_u1_normalized_pred[:-1])**2, dim=0))
cg_model.sigma_v = torch.sqrt(torch.mean((train_v[1:] - train_v_pred[:-1])**2, dim=0))
cg_model.sigma_u2 = torch.sqrt(torch.mean((train_u2_normalized[1:] - train_u2_normalized_pred[:-1])**2, dim=0))



# Training Stage 2.1: Train CGModel with loss_ae + loss_forecast_u + loss_forecast_z + loss_DA(MSE)
epochs = 1000
batch_size = 1000
batch_size_DA = 5
seq_len_DA = 100
warm_out_DA = 20
train_tensor = torch.utils.data.TensorDataset(train_u_normalized[:-1], train_u_normalized[1:])
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history_ae = []
train_loss_history_forecast_u = []
train_loss_history_forecast_v = []
train_loss_history_DA_mean = []

optimizer = torch.optim.Adam(cg_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    start_time = time.time()

    train_loss_ae = 0.
    train_loss_forecast_u = 0.
    train_loss_forecast_v = 0.
    train_loss_DA_mean = 0.
    for u, u_next in train_loader:
        u, u_next = u.to(device), u_next.to(device)

        # AutoEncoder
        u2 = u[:, indices_u2]
        v = cg_model.autoencoder.encoder(u2)
        u2_ae = cg_model.autoencoder.decoder(v)
        loss_ae = nnF.mse_loss(u2, u2_ae)

        #  State Forecast
        u1 = u[:, indices_u1]
        uCG = torch.cat([u1, v], dim=-1)
        uCG_pred = cg_model.forecast(uCG)
        u1_pred = uCG_pred[:, :dim_u1]
        v_pred = uCG_pred[:, dim_u1:]
        u2_pred = cg_model.autoencoder.decoder(v_pred)
        u_pred = torch.zeros_like(u)
        u_pred[:, indices_u1] = u1_pred
        u_pred[:, indices_u2] = u2_pred
        loss_forecast_u = nnF.mse_loss(u_next, u_pred)

        u2_next = u_next[:, indices_u2]
        v_next = cg_model.autoencoder.encoder(u2_next)
        loss_forecast_v = nnF.mse_loss(v_next, v_pred)

        # DA
        indices_init = np.random.choice(Ntrain - seq_len_DA + 1, size=batch_size_DA, replace=False)
        u_seq = torch.stack([train_u_normalized[i:i+seq_len_DA] for i in indices_init]).to(device)
        u1_seq = u_seq[:, :, indices_u1]
        u2_seq = u_seq[:, :, indices_u2]
        mu_DA_v_seq, cov_DA_v_seq = cg_model.filter(u1_seq)
        mu_DA_u2_seq = cg_model.decode_mean(mu_DA_v_seq.reshape(-1, dim_v), cov_DA_v_seq.reshape(-1, dim_v, dim_v), u1_seq.reshape(-1, dim_u1)).reshape(batch_size_DA, seq_len_DA, dim_u2)
        loss_da = nnF.mse_loss(mu_DA_u2_seq[:, warm_out_DA:], u2_seq[:, warm_out_DA:])

        loss_total = loss_ae + loss_forecast_u + loss_forecast_v + 0.2*loss_da

        loss_total.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss_ae += loss_ae.item()
        train_loss_forecast_u += loss_forecast_u.item()
        train_loss_forecast_v += loss_forecast_v.item()
        train_loss_DA_mean += loss_da.item()
    train_loss_ae /= train_num_batches
    train_loss_forecast_u /= train_num_batches
    train_loss_forecast_v /= train_num_batches
    train_loss_DA_mean /= train_num_batches
    train_loss_history_ae.append(train_loss_ae)
    train_loss_history_forecast_u.append(train_loss_forecast_u)
    train_loss_history_forecast_v.append(train_loss_forecast_v)
    train_loss_history_DA_mean.append(train_loss_DA_mean)

    end_time = time.time()
    print(f"ep {ep} | "
          f"time: {end_time - start_time:.4f} | "
          f"loss AE: {train_loss_ae:.4f} | "
          f"loss u: {train_loss_forecast_u:.4f} | "
          f"loss v: {train_loss_forecast_v:.4f} | "
          f"loss DA: {train_loss_DA_mean:.4f}")
"""

# torch.save(cg_model.state_dict(), r"./model/L96_CGM_v80_stage2_1.pt")
# np.save(r"./model/L96_CGM_v80_train_loss_history_AE_stage2_1.npy", train_loss_history_ae)
# np.save(r"./model/L96_CGM_v80_train_loss_history_forecast_u_stage2_1.npy", train_loss_history_forecast_u)
# np.save(r"./model/L96_CGM_v80_train_loss_history_forecast_v_stage2_1.npy", train_loss_history_forecast_v)
# np.save(r"./model/L96_CGM_v80_train_loss_history_DA_mean_stage2_1.npy", train_loss_history_DA_mean)

cg_model.load_state_dict(torch.load(r"./model/L96_CGM_v80_stage2_1.pt"))


# Training Stage 2.2: Train CGModel with loss_DA (NLL)

def nll_loss(x, mu, cov):
    # x: (N, S),  mu: (N, S),  cov: (N, S)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    nll = 0.5*(((x-mu)**2/cov + torch.log(cov)).sum(dim=-1) + x.shape[-1]*torch.log(2*pi))
    return nll.mean()

seq_len = Ntrain
train_u1_normalized_batched = train_u1_normalized.reshape(-1, seq_len, dim_u1)
batch_size = train_u1_normalized_batched.shape[0]
with torch.no_grad():
    train_mu_DA_v_batched, train_cov_DA_v_batched = cg_model.filter(train_u1_normalized_batched)
train_mu_DA_v_batched_flatten = train_mu_DA_v_batched.reshape(-1, dim_v)
train_cov_DA_v_batched_flatten = train_cov_DA_v_batched.reshape(-1, dim_v, dim_v)
with torch.no_grad():
    train_mu_DA_u2_normalized_batched_flatten = cg_model.decode_mean(train_mu_DA_v_batched_flatten, train_cov_DA_v_batched_flatten, train_u1_normalized)

epochs = 100
batch_size = 2000
train_tensor = torch.utils.data.TensorDataset(train_mu_DA_v_batched_flatten, train_cov_DA_v_batched_flatten, train_u1_normalized, train_mu_DA_u2_normalized_batched_flatten, train_u2_normalized)
train_loader = torch.utils.data.DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
train_num_batches = len(train_loader)
Niters = epochs * train_num_batches
train_loss_history_DA_cov = []

for p in cg_model.parameters():
    p.requires_grad_(False)
for p in cg_model.Gdecoder.cov_decoder.parameters():
    p.requires_grad_(True)
optimizer = torch.optim.Adam(cg_model.Gdecoder.cov_decoder.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Niters)
"""
for ep in range(1, epochs+1):
    train_loss_DA_cov = 0.
    start_time = time.time()
    for mu_v, cov_v, u1, mu_u2, u2 in train_loader:
        cov_u2 = cg_model.decode_cov(mu_v, cov_v, u1)
        loss = nll_loss(u2, mu_u2, cov_u2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss_DA_cov += loss.item()
    train_loss_DA_cov /= train_num_batches
    train_loss_history_DA_cov.append(train_loss_DA_cov)

    end_time = time.time()
    print(f"ep {ep} | "
          f"time: {end_time - start_time:.4f} | "
          f"loss DA cov: {train_loss_DA_cov:.4f}")
"""

# torch.save(cg_model.state_dict(), r"./model/L96_CGM_v80_stage2_2.pt")
# np.save(r"./model/L96_CGM_v80_train_loss_history_DA_cov_stage2_2.npy", train_loss_history_DA_cov)

cg_model.load_state_dict(torch.load(r"./model/L96_CGM_v80_stage2_2.pt"))


#################################################
################# Model Inference ###############
#################################################

# Ground Truth (Data without Noisy Observations)
u_true = torch.tensor(np.load("data/SimulationData/L96_Data.npy"), dtype=torch.float).to(device)
u_true = torch.cat([u_true[:, indices_obs], u_true], dim=-1)
train_u_true = u_true[:Ntrain]
test_u_true = u_true[-Ntest:]
train_u1_true = train_u_true[:, indices_u1]
test_u1_true = test_u_true[:, indices_u1]
train_u2_true = train_u_true[:, indices_u2]
test_u2_true = test_u_true[:, indices_u2]


# State Forecasting
with torch.no_grad():
    train_v = cg_model.autoencoder.encoder(train_u2_normalized)
    train_uCG = torch.cat([train_u1_normalized, train_v], dim=-1)
    train_uCG_pred = cg_model.forecast(train_uCG)
    train_u1_normalized_pred = train_uCG_pred[:, :dim_u1]
    train_v_pred = train_uCG_pred[:, dim_u1:]
    train_u2_normalized_pred = cg_model.autoencoder.decoder(train_v_pred)
    train_u_normalized_pred = torch.zeros_like(train_u_normalized)
    train_u_normalized_pred[:, indices_u1] = train_u1_normalized_pred
    train_u_normalized_pred[:, indices_u2] = train_u2_normalized_pred
    train_u_pred = Normalizer.decode(train_u_normalized_pred)
train_forecast_RMSE = RMSE(train_u_true[1:, indices_u2], train_u_pred[:-1, indices_u2])
print("train forecast RMSE: ", train_forecast_RMSE)

with torch.no_grad():
    test_v = cg_model.autoencoder.encoder(test_u2_normalized)
    test_uCG = torch.cat([test_u1_normalized, test_v], dim=-1)
    test_uCG_pred = cg_model.forecast(test_uCG)
    test_u1_normalized_pred = test_uCG_pred[:, :dim_u1]
    test_v_pred = test_uCG_pred[:, dim_u1:]
    test_u2_normalized_pred = cg_model.autoencoder.decoder(test_v_pred)
    test_u_normalized_pred = torch.zeros_like(test_u_normalized)
    test_u_normalized_pred[:, indices_u1] = test_u1_normalized_pred
    test_u_normalized_pred[:, indices_u2] = test_u2_normalized_pred
    test_u_pred = Normalizer.decode(test_u_normalized_pred)
test_forecast_RMSE = RMSE(test_u_true[1:, indices_u2], test_u_pred[:-1, indices_u2])
print("test forecast RMSE: ", test_forecast_RMSE)


# State Estimation
start_time = time.time()
with torch.no_grad():
    train_mu_DA_v, train_cov_DA_v = cg_model.filter(train_u1_normalized[None])
    train_mu_DA_u2_normalized = cg_model.decode_mean(train_mu_DA_v[0], train_cov_DA_v[0], train_u1_normalized)
    train_cov_DA_u2_normalized = cg_model.decode_cov(train_mu_DA_v[0], train_cov_DA_v[0], train_u1_normalized)
train_mu_DA_u2 = Normalizer.std[indices_u2] * train_mu_DA_u2_normalized +  Normalizer.mean[indices_u2]
train_cov_DA_u2 = Normalizer.std[indices_u2]**2 * train_cov_DA_u2_normalized
end_time = time.time()
end_time - start_time
train_DA_RMSE = RMSE(train_u2_true[:, indices_u2_unobs], train_mu_DA_u2[:, indices_u2_unobs])
train_DA_NLL = NLL(train_u2_true[:, indices_u2_unobs], train_mu_DA_u2[:, indices_u2_unobs], train_cov_DA_u2[:, indices_u2_unobs])
print("train DA RMSE: ", train_DA_RMSE)
print("train DA NLL: ", train_DA_NLL)

start_time = time.time()
with torch.no_grad():
    test_mu_DA_v, test_cov_DA_v = cg_model.filter(test_u1_normalized[None])
    test_mu_DA_u2_normalized = cg_model.decode_mean(test_mu_DA_v[0], test_cov_DA_v[0], test_u1_normalized)
    test_cov_DA_u2_normalized = cg_model.decode_cov(test_mu_DA_v[0], test_cov_DA_v[0], test_u1_normalized)
test_mu_DA_u2 = Normalizer.std[indices_u2] * test_mu_DA_u2_normalized +  Normalizer.mean[indices_u2]
test_cov_DA_u2 = Normalizer.std[indices_u2]**2 * test_cov_DA_u2_normalized
end_time = time.time()
end_time - start_time
test_DA_RMSE = RMSE(test_u2_true[:, indices_u2_unobs], test_mu_DA_u2[:, indices_u2_unobs])
test_DA_NLL = NLL(test_u2_true[:, indices_u2_unobs], test_mu_DA_u2[:, indices_u2_unobs], test_cov_DA_u2[:, indices_u2_unobs])
print("test DA RMSE: ", test_DA_RMSE)
print("test DA NLL: ", test_DA_NLL)


num_params = sum(p.numel() for p in cg_model.parameters())
print("num params: ", num_params)


#####################################################
################# Visualization Data ################
#####################################################

# Data Assimilation Result with Observed Data
test_mu_DA = test_u2.clone()
test_std_DA = torch.zeros_like(test_u2)
test_mu_DA[:, indices_u2_unobs] = test_mu_DA_u2[:, indices_u2_unobs]
test_std_DA[:, indices_u2_unobs] = test_cov_DA_u2[:, indices_u2_unobs]**0.5
# np.savez(r"./data/VisualizationData/L96_DA_CGM.npz", mean=test_mu_DA, std=test_std_DA)

# Multiple-Step State Forecasting
def forecast_next_state(u):
    u_normalized = Normalizer.encode(u)
    u1_normalized = u_normalized[:, indices_u1]
    u2_normalized = u_normalized[:, indices_u2]
    Nt = u_normalized.shape[0]
    dim_u1 = u1_normalized.shape[1]
    dim_u2 = u2_normalized.shape[1]
    with torch.no_grad():
        v = cg_model.autoencoder.encoder(u2_normalized)
        uCG = torch.cat([u1_normalized, v], dim=-1)
        uCG_pred = cg_model.forecast(uCG)
        u1_normalized_pred = uCG_pred[:, :dim_u1]
        v_pred = uCG_pred[:, dim_u1:]
        u2_normalized_pred = cg_model.autoencoder.decoder(v_pred)
        u2_normalized_pred[:, indices_obs] = u1_normalized_pred
        u_normalized_pred = torch.zeros(Nt, dim_u1+dim_u2, device=u1_normalized.device)
        u_normalized_pred[:, indices_u1] = u1_normalized_pred
        u_normalized_pred[:, indices_u2] = u2_normalized_pred
        u_pred = Normalizer.decode(u_normalized_pred)
    return u_pred
pred_step = 100
test_u_multistep_pred = torch.zeros(Ntest, pred_step+1, dim_u1+dim_u2, device=device)
test_u_multistep_pred[:, 0] = test_u
for n in range(pred_step):
    print("Forecast Step:", n+1)
    test_u_multistep_pred[:, n+1] = forecast_next_state(test_u_multistep_pred[:, n])
# np.save(r"./data/VisualizationData/L96_Forecast_CGM.npy", test_u_multistep_pred[:, :, indices_u2])

memory = pred_step+1
test_u_multistep_pred = test_u_multistep_pred[:-memory+1]
test_rollout_true = test_u_true.unfold(dimension=0, size=memory, step=1).permute(0, 2, 1)
RMSE(test_u_multistep_pred, test_rollout_true)

