# Stochastic Lorenz 63 System with Observation Noise
import numpy as np

####################################################
################# Data Preparation #################
####################################################

# Simulation Settings: Lt=1000, dt=0.001
sigma, rho, beta = 10, 28, 8/3
sigma_dyn = 0.5
Lt = 2000
dt_sim = 0.001
Nt_sim = int(Lt/dt_sim)
u = np.zeros((Nt_sim, 3))
for n in range(Nt_sim-1):
    if n % (1/dt_sim) == 0:
        print(int(n*dt_sim))
    u[n+1, 0] = u[n, 0] + (sigma*(u[n, 1] - u[n, 0]))*dt_sim
    u[n+1, 1] = u[n, 1] + (u[n, 0]*(rho-u[n, 2])-u[n, 1])*dt_sim
    u[n+1, 2] = u[n, 2] + (u[n, 0]*u[n, 1] - beta*u[n, 2])*dt_sim
    u[n+1] = u[n+1] + dt_sim**0.5*sigma_dyn*np.random.randn(*u[n+1].shape)
u = u[::50] # dt_obs = 0.05

# Measurement Noise
sigma_obs = 2.
u_obs = u + sigma_obs*np.random.randn(*u.shape)

# np.save("./data/L63_Data.npy", u)
# np.save("./data/L63_Data_NoisyObs.npy", u_obs)


