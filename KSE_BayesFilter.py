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


##########################################################
################# State-Space Model ######################
##########################################################

def dynamics_KSE(u):
    # u shape: (N, state_dim)
    Lx = 22.0
    Nx = u.shape[-1]
    device = u.device
    dtype = u.dtype
    kx = torch.cat([ torch.arange(0, Nx // 2, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype), torch.arange(-Nx // 2 + 1, 0, device=device, dtype=dtype) ])
    alpha = 2.0 * math.pi * kx / Lx
    D = 1j * alpha
    L = alpha**2 - alpha**4
    G = -0.5 * D
    u_hat = torch.fft.fft(u, dim=-1)
    linear = L * u_hat
    nonlinear = G * torch.fft.fft(u**2, dim=-1)
    u_dot = torch.real(torch.fft.ifft(linear + nonlinear, dim=-1))
    return u_dot

def forecast_KSE(u, delta_t=dt_obs):
    Lx = 22.0
    dt_sim = 0.025
    Nt_sim = int(delta_t / dt_sim)
    Nx = u.shape[-1]
    device = u.device
    dtype = u.dtype
    # Fourier wavenumbers, matching the NumPy code
    kx = torch.cat([ torch.arange(0, Nx // 2, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype), torch.arange(-Nx // 2 + 1, 0, device=device, dtype=dtype)])
    alpha = 2.0 * math.pi * kx / Lx
    D = 1j * alpha
    L = alpha**2 - alpha**4
    G = -0.5 * D
    A = 1.0 + dt_sim / 2.0 * L
    B = 1.0 / (1.0 - dt_sim / 2.0 * L)
    # Transform current state to Fourier space
    un = torch.fft.fft(u, dim=-1)
    # Initial nonlinear term: N(u) = -1/2 d/dx(u^2)
    Nn = G * torch.fft.fft(u**2, dim=-1)
    for _ in range(Nt_sim):
        Nn1 = Nn
        u_phys = torch.real(torch.fft.ifft(un, dim=-1))
        Nn = G * torch.fft.fft(u_phys**2, dim=-1)
        # CNAB2 update
        un = B * (A * un + 1.5 * dt_sim * Nn - 0.5 * dt_sim * Nn1)
    # Back to physical space
    u_next = torch.real(torch.fft.ifft(un, dim=-1))
    return u_next

def forecast_SKSE(u, delta_t=dt_obs):
    Lx = 22.0
    dt_sim = 0.025
    Nt_sim = int(delta_t / dt_sim)
    Nx = u.shape[-1]
    device = u.device
    dtype = u.dtype
    # Fourier wavenumbers, matching the NumPy code
    kx = torch.cat([ torch.arange(0, Nx // 2, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype), torch.arange(-Nx // 2 + 1, 0, device=device, dtype=dtype)])
    alpha = 2.0 * math.pi * kx / Lx
    D = 1j * alpha
    L = alpha**2 - alpha**4
    G = -0.5 * D
    A = 1.0 + dt_sim / 2.0 * L
    B = 1.0 / (1.0 - dt_sim / 2.0 * L)
    # Transform current state to Fourier space
    un = torch.fft.fft(u, dim=-1)
    # Initial nonlinear term: N(u) = -1/2 d/dx(u^2)
    Nn = G * torch.fft.fft(u**2, dim=-1)
    sigma_dyn = 5. # Hyper-parameter
    for _ in range(Nt_sim):
        Nn1 = Nn
        u_phys = torch.real(torch.fft.ifft(un, dim=-1))
        Nn = G * torch.fft.fft(u_phys**2, dim=-1)
        # CNAB2 update
        un = B * (A * un + 1.5 * dt_sim * Nn - 0.5 * dt_sim * Nn1)
        # Add stochastic dynamics noise
        u_phys = torch.real(torch.fft.ifft(un, dim=-1))
        u_phys = u_phys + sigma_dyn * dt_sim**0.5 * torch.randn_like(u_phys)
        un = torch.fft.fft(u_phys, dim=-1)
    # Back to physical space
    u_next = torch.real(torch.fft.ifft(un, dim=-1))
    return u_next

def obs_map_KSE(u):
    # u: (N, S)
    return u[:, indices_u1]

def Jacobian_dyn_KSE(u):
    # u: (B, state_dim)
    # return: (B, state_dim, state_dim)

    Lx = 22.0
    B, Nx = u.shape
    device = u.device
    dtype = u.dtype

    kx = torch.cat([ torch.arange(0, Nx // 2, device=device, dtype=dtype), torch.zeros(1, device=device, dtype=dtype), torch.arange(-Nx // 2 + 1, 0, device=device, dtype=dtype) ])

    alpha = 2.0 * math.pi * kx / Lx
    D = 1j * alpha
    L = alpha**2 - alpha**4
    G = -0.5 * D

    eye = torch.eye(Nx, device=device, dtype=dtype)  # (Nx, Nx)

    # Linear part: v -> real(ifft(L * fft(v)))
    linear_basis = torch.real( torch.fft.ifft(L * torch.fft.fft(eye, dim=-1), dim=-1) )  # (basis_index, output_index)

    # Nonlinear part: v -> real(ifft(G * fft(2*u*v)))
    perturb = 2.0 * u[:, None, :] * eye[None, :, :]  # (B, basis_index, state_dim)

    nonlinear_basis = torch.real( torch.fft.ifft(G * torch.fft.fft(perturb, dim=-1), dim=-1) )  # (B, basis_index, output_index)

    # Convert from (basis_index, output_index) to (output_index, input_index)
    J = linear_basis.T[None, :, :] + nonlinear_basis.transpose(1, 2)
    return J

def Jacobian_obs_KSE(u):
    # u: (B, state_dim)
    # return: (B, obs_dim, state_dim)

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
u_true = torch.tensor(np.load("data/SimulationData/KSE_Data.npy"), dtype=torch.float)
u_true = u_true[::4, ::16]
train_state_true = u_true[:Ntrain]
test_state_true = u_true[-Ntest:]
train_obs_true = u_true[:Ntrain, indices_u1]
test_obs_true = u_true[-Ntest:, indices_u1]


# EKF
def EKF( y, state_dim, forecast_dyn, obs_map, cov_dyn, cov_obs, dt_obs, dt_sim, Jacobian_dyn=None, Jacobian_obs=None, mu0=None, cov0=None, ):
    device = y.device
    dtype = y.dtype
    Nt, obs_size = y.shape
    Nt_sim = int(round(dt_obs / dt_sim))

    I = torch.eye(state_dim, device=device, dtype=dtype)

    cov_dyn = cov_dyn.to(device=device, dtype=dtype)
    cov_obs = cov_obs.to(device=device, dtype=dtype)

    mu_post = torch.zeros( Nt + 1, state_dim, device=device, dtype=dtype )
    cov_post = torch.zeros( Nt + 1, state_dim, state_dim, device=device, dtype=dtype, )

    cov_post[0] = I

    if mu0 is not None:
        mu_post[0] = mu0.to(device=device, dtype=dtype)

    if cov0 is not None:
        cov_post[0] = cov0.to(device=device, dtype=dtype)

    def forecast_single(mu):
        return forecast_dyn(mu[None, :])[0]

    def obs_single(mu):
        return obs_map(mu[None, :])[0]

    def jac_dyn_single(mu):
        return Jacobian_dyn(mu[None, :])[0]

    def jac_obs_single(mu):
        return Jacobian_obs(mu[None, :])[0]

    # =========================================================
    # Added for KSE:
    # Construct the spectral operators used by the semi-implicit
    # CNAB2 mean integrator. This replaces explicit Euler for the
    # EKF mean because explicit Euler is unstable for the stiff
    # fourth-order spatial derivative in KSE.
    # =========================================================
    Lx = 22.0
    Nx = state_dim
    kx = torch.cat([ torch.arange( 0, Nx // 2, device=device, dtype=dtype ), torch.zeros(1, device=device, dtype=dtype), torch.arange( -Nx // 2 + 1, 0, device=device, dtype=dtype, ), ])
    alpha = 2.0 * math.pi * kx / Lx
    # Linear Fourier operator: -u_xx - u_xxxx
    L_kse = alpha**2 - alpha**4

    # Nonlinear Fourier operator: -0.5*d_x(u^2)
    G_kse = -0.5j * alpha

    # Crank-Nicolson coefficients for the linear term
    A_kse = 1.0 + 0.5 * dt_sim * L_kse
    B_kse = 1.0 / (1.0 - 0.5 * dt_sim * L_kse)

    def nonlinear_hat_kse(mu):
        return G_kse * torch.fft.fft(mu**2, dim=-1)

    # End of added KSE spectral setup
    # =========================================================
    for nt in range(Nt):
        print(f"EKF step {nt + 1}")
        # 1) Forecast
        mu = mu_post[nt]
        cov = cov_post[nt]
        # =====================================================
        # Added for KSE:
        # Initialize the CNAB2 mean forecast at the beginning
        # of each observation interval. The nonlinear history
        # is restarted because the EKF analysis changes the mean
        # discontinuously.
        # =====================================================
        mu_hat = torch.fft.fft(mu, dim=-1)
        nonlinear_previous = nonlinear_hat_kse(mu)
        # End of added CNAB2 initialization
        # =====================================================
        for _ in range(Nt_sim):
            if Jacobian_dyn is not None:
                F = jac_dyn_single(mu)
            else:
                F = torch.autograd.functional.jacobian( forecast_single, mu.detach().requires_grad_(True), )
            # =================================================
            # Added for KSE:
            # Semi-implicit CNAB2 mean update. The linear stiff
            # term is handled implicitly, while the nonlinear
            # term is handled by Adams-Bashforth.
            #
            # This replaces:
            # mu = mu + forecast_single(mu) * dt_sim
            # =================================================
            nonlinear_current = nonlinear_hat_kse(mu)
            mu_hat = B_kse * ( A_kse * mu_hat + dt_sim * ( 1.5 * nonlinear_current - 0.5 * nonlinear_previous ) )
            nonlinear_previous = nonlinear_current
            mu = torch.fft.ifft( mu_hat, dim=-1 ).real
            # End of added KSE mean update
            # =================================================
            # Covariance forecast
            A = torch.matrix_exp(F * dt_sim)
            cov = ( A @ cov @ A.T + cov_dyn * dt_sim )
            cov = ( 0.5 * (cov + cov.T) + 1e-7 * I )

        if Jacobian_obs is not None:
            H = jac_obs_single(mu)
        else:
            H = torch.autograd.functional.jacobian( obs_single, mu.detach().requires_grad_(True), )

        mu_y = obs_single(mu)

        # 2) Analysis
        cov_yy = H @ cov @ H.T + cov_obs
        cov_yy = cov_yy + 1e-9 * torch.eye( obs_size, device=device, dtype=dtype, )
        cov_xy = cov @ H.T
        K = torch.linalg.solve( cov_yy.T, cov_xy.T, ).T
        mu_post[nt + 1] = ( mu + K @ (y[nt] - mu_y) )
        # Joseph covariance update
        KH = K @ H
        cov = ( (I - KH) @ cov @ (I - KH).T + K @ cov_obs @ K.T )
        cov = ( 0.5 * (cov + cov.T) + 1e-9 * I )
        cov_post[nt + 1] = cov
    return mu_post[1:], cov_post[1:]

torch.manual_seed(0)
np.random.seed(0)
mu0 = train_state.mean(dim=0)
cov0 = torch.diag(train_state.var(dim=0, unbiased=True))
cov_dyn = 0.5**2*torch.eye(state_dim).to(device)  # Tunable Parameter / Dynamics Noise Injection
obs_inflation = 15
st = time.time()
mu_EKF, cov_EKF = EKF(y=test_obs, state_dim=state_dim, forecast_dyn=dynamics_KSE, obs_map=obs_map_KSE, cov_dyn=cov_dyn, cov_obs=(obs_inflation*sigma_obs)**2 * torch.eye(obs_dim), dt_obs=dt_obs, dt_sim=0.025, Jacobian_dyn=Jacobian_dyn_KSE, Jacobian_obs=Jacobian_obs_KSE, mu0=mu0, cov0=cov0)
et = time.time()
print("EKF metrics: ", metrics_unobs(test_state_true, mu_EKF, cov_EKF))
print("EKF time: ", et - st)
# np.savez(r"./data/VisualizationData/KSE_DA_EKF.npz", mean=mu_EKF, std=cov_EKF.diagonal(dim1=-2, dim2=-1)**0.5)


# UKF
def UKF( y, state_dim, forecast, obs_map, cov_dyn, cov_obs, dt_obs, dt_sim, alpha, k, beta=2, mu0=None, cov0=None ):
    def moment2sigma(mu, cov):
        sigma_pts = torch.zeros( 2 * state_dim + 1, state_dim, device=device )
        S = torch.linalg.cholesky(c * cov)
        sigma_pts[0] = mu
        for i in range(state_dim):
            sigma_pts[i + 1] = mu + S[:, i]
            sigma_pts[i + state_dim + 1] = mu - S[:, i]
        return sigma_pts
    def sigma2moment(sigma_pts):
        mu = w_mean @ sigma_pts
        cov = ( (sigma_pts - mu).T * w_cov ) @ (sigma_pts - mu)
        return mu, cov
    device = y.device
    Nt, obs_size = y.shape
    Nt_sim = int(dt_obs / dt_sim)
    I = torch.eye(state_dim, device=device)
    lam = alpha**2 * (state_dim + k) - state_dim
    c = state_dim + lam
    w_mean = ( 1 / (2 * c) * torch.ones( 2 * state_dim + 1, device=device ) )
    w_cov = ( 1 / (2 * c) * torch.ones( 2 * state_dim + 1, device=device ) )
    w_mean[0] = lam / c
    w_cov[0] = lam / c + (1 - alpha**2 + beta)
    mu_post = torch.zeros( Nt + 1, state_dim, device=device )
    cov_post = I.unsqueeze(0).repeat( Nt + 1, 1, 1 )
    if mu0 is not None:
        mu_post[0] = mu0
    if cov0 is not None:
        cov_post[0] = cov0
    for nt in range(Nt):
        print(rf"UKF step {nt + 1}")
        # 1) Forecast
        sigma_pts = moment2sigma(mu_post[nt], cov_post[nt])
        # MODIFIED FOR KSE:
        # Propagate all sigma points over the complete observation interval.
        # This preserves the CNAB2 nonlinear history inside forecast_KSE.
        sigma_pts = forecast(sigma_pts, delta_t=dt_obs)
        mu, cov = sigma2moment(sigma_pts)
        # MODIFIED FOR KSE:
        # Add the process covariance over the complete observation interval.
        cov = cov + cov_dyn * dt_obs
        sigma_pts = moment2sigma(mu, cov)
        gamma_pts = obs_map(sigma_pts)
        # 2) Analysis
        mu_prior, cov_prior = sigma2moment(sigma_pts)
        mu_y = w_mean @ gamma_pts
        cov_yy = ( ((gamma_pts - mu_y).T * w_cov) @ (gamma_pts - mu_y) + cov_obs )
        ccov_xy = ( ((sigma_pts - mu_prior).T * w_cov) @ (gamma_pts - mu_y) )
        K = torch.linalg.solve( cov_yy.T, ccov_xy.T ).T
        mu_post[nt + 1] = ( mu_prior + K @ (y[nt] - mu_y) )
        cov_post[nt + 1] = ( cov_prior - K @ ccov_xy.T )
    return mu_post[1:], cov_post[1:]

torch.manual_seed(0)
np.random.seed(0)
mu0 = train_state.mean(dim=0)
cov0 = torch.diag(train_state.var(dim=0, unbiased=True))
cov_dyn = 0.5**2*torch.eye(state_dim).to(device)  # Tunable Parameter / Dynamics Noise Injection
st = time.time()
mu_UKF, cov_UKF = UKF(y=test_obs, state_dim=state_dim, forecast=forecast_KSE, obs_map=obs_map_KSE, cov_dyn=cov_dyn, cov_obs=sigma_obs**2 * torch.eye(obs_dim), dt_obs=dt_obs, dt_sim=0.025, alpha=0.6, k=0, beta=2, mu0=mu0, cov0=cov0)
et = time.time()
print("UKF metrics: ", metrics_unobs(test_state_true, mu_UKF, cov_UKF))
print("UKF time: ", et - st)
# np.savez(r"./data/VisualizationData/KSE_DA_UKF.npz", mean=mu_UKF, std=cov_UKF.diagonal(dim1=-2, dim2=-1)**0.5)


# EnKF
torch.manual_seed(0)
np.random.seed(0)
def GC(gridx, gridy, loc_radius, L):
    d = np.abs(gridx[:, None] - gridy[None, :])
    d = np.minimum(d, L - d)  # periodic_dist
    r = d / loc_radius
    rho = np.zeros_like(r)
    m1 = (r <= 1.0)
    m2 = (r > 1.0) & (r <= 2.0)
    r1 = r[m1]
    r2 = r[m2]
    # 0 <= r <= 1
    rho[m1] = 1.0 - (5.0/3.0) * r1**2 + (5.0/8.0) * r1**3 + 0.5 * r1**4 - 0.25 * r1**5
    # 1 < r <= 2
    rho[m2] = 4.0 - 5.0 * r2 + (5.0/3.0) * r2**2 + (5.0/8.0) * r2**3 - 0.5 * r2**4 + (1.0/12.0) * r2**5 - 2.0 / (3.0 * r2)
    return rho
GC_xy = torch.tensor(GC(grid_state, grid_obs, 22., Lx), dtype=torch.float)
ens_size = 500
x0 = train_state[np.random.randint(0, Ntrain, size=ens_size)]
process_noise_coef = 0.2
st = time.time()
x_ens = EnKF(y=test_obs, state_dim=state_dim, forecast=forecast_KSE, obs_map=obs_map_KSE, cov_obs=sigma_obs**2*torch.eye(obs_dim), J=ens_size, x0=x0, process_noise_coef=process_noise_coef)
et = time.time()
mu_EnKF = x_ens.mean(dim=0)
# cov_EnKF = torch.einsum("nti,ntj->tij", x_ens-mu_EnKF, x_ens-mu_EnKF) / (x_ens.shape[0]-1)
cov_EnKF = x_ens.var(dim=0, unbiased=True)
print("EnKF metrics: ", metrics_unobs(test_state_true, mu_EnKF, cov_EnKF))
print("EnKF time: ", et - st)
# np.savez(r"./data/VisualizationData/KSE_DA_EnKF.npz", mean=mu_EnKF, std=cov_EnKF**0.5)


# PF
torch.manual_seed(0)
np.random.seed(0)
particle_size = 2000
likelihood_inflation = 5.
roughening = 0.01
ess_frac = 0.7
process_noise_coef = 0.1
x0 = train_state[np.random.randint(0, Ntrain, size=particle_size)]
st = time.time()
mu_PF, cov_PF = PF(y=test_obs, state_dim=state_dim, forecast=forecast_KSE, obs_map=obs_map_KSE, cov_obs=(likelihood_inflation*sigma_obs)**2*torch.eye(obs_dim), J=particle_size, x0=x0, ess_frac=ess_frac, process_noise_coef=process_noise_coef, roughening=roughening)
et = time.time()
print("PF metrics: ", metrics_unobs(test_state_true, mu_PF, cov_PF))
print("PF time: ", et - st)
# np.savez(r"./data/VisualizationData/KSE_DA_PF.npz", mean=mu_PF, std=cov_PF**0.5)
