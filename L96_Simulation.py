# Stochastic Lorenz 96 System with Observation Noise
import numpy as np

F = 8
sigma_dyn = 0.5

I = 40
Lt = 5000
dt_sim = 0.001
Nt_sim = int(Lt/dt_sim)
t = np.linspace(0, Lt, Nt_sim)
u = np.zeros((Nt_sim, I))
for n in range(Nt_sim-1):
    if n % (1/dt_sim) == 0:
        print(int(n*dt_sim))
    for i in range(I):
        u_dot = -u[n, i] + u[n,(i+1)%I]*u[n,i-1] - u[n,i-2]*u[n,i-1] + F
        u[n+1, i] = u[n, i] + u_dot*dt_sim + sigma_dyn*dt_sim**0.5*np.random.randn()
u = u[::50]  # dt_obs = 0.05


# Measurement Noise
sigma_obs = 0.75
u_obs = u + sigma_obs*np.random.randn(*u.shape)


np.save("data/SimulationData/L96_Data.npy", u)
np.save("data/SimulationData/L96_Data_NoisyObs.npy", u_obs)

