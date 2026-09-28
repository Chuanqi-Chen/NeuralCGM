import numpy as np
import scipy as sp

#Simulation Setting: Lx = 22, Lt = 5000, dx=22/2048, dt = 0.025
Lx = 22
Nx = 2048
x = Lx*np.arange(0, Nx, 1)/Nx
Lt = 50000
dt = 0.025
Nt = int(Lt/dt)
nplot = 10
Nplot = int(Nt/nplot) + 1  # total number of saved time steps
t = np.arange(0,Nplot,1)*dt*nplot
u0 = 0.1*np.cos(x/16.)*(1.+2.*np.sin(x/16.))
U = np.zeros((Nplot, Nx))
U[0,:] = u0 # save initial value u to matrix U

kx = np.hstack([np.arange(0,Nx/2,1), np.array([0]), np.arange(-Nx/2+1,0,1)])  # integer wavenumbers: exp(2*pi*i*kx*x/L)
alpha = 2.*np.pi*kx/Lx              # real wavenumbers:    exp(i*alpha*x)
D = 1j*alpha                   # D = d/dx operator in Fourier space
L = alpha**2 - alpha**4        # linear operator -D^2 - D^4 in Fourier space
G = -0.5*D                      # -1/2 D operator in Fourier space
A =  np.ones(Nx) + dt/2*L
B = 1/(np.ones(Nx) - dt/2*L)

counter = 1     # counter for saved data
Nn  = sp.fftpack.fft(u0**2)*G  # -1/2 d/dx(u^2) = -u u_x, collocation calculation
un = sp.fftpack.fft(u0) # transform data to spectral coeffs
for n in range(Nt):
    Nn1 = Nn     # shift N^{n-1} <- N^n
    Nn  = G * sp.fftpack.fft(np.real(sp.fftpack.ifft(un)) ** 2) # compute N^n = -u u_x
    un = B * (A * un + 3/2*dt * Nn - dt/2 * Nn1) # CNAB2 formula
    if (n+1) % nplot == 0:
        U[counter, :] = np.real(sp.fftpack.ifft(un))
        counter += 1

noise_level = 0.15
sigma_obs = noise_level * np.std(U)
U_obs = U + sigma_obs * np.random.randn(*U.shape)

U_obs  = U_obs[:-1]
U = U[:-1]
t = t[:-1]

# np.save(r"./data/KSE_Data.npy", U)
# np.save(r"./data/KSE_Data_NoisyObs.npy", U_obs)
