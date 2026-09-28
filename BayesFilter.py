import torch

def EKF(y, state_dim, forecast_dyn, obs_map, cov_dyn, cov_obs, dt_obs, dt_sim, Jacobian_dyn=None, Jacobian_obs=None, mu0=None, cov0=None):
    device = y.device
    dtype = y.dtype
    Nt, obs_size = y.shape
    Nt_sim = int(dt_obs / dt_sim)

    I = torch.eye(state_dim, device=device, dtype=dtype)

    mu_post = torch.zeros(Nt + 1, state_dim, device=device, dtype=dtype)
    cov_post = torch.zeros(Nt + 1, state_dim, state_dim, device=device, dtype=dtype)
    cov_post[0] = I
    if mu0 is not None:
        mu_post[0] = mu0
    if cov0 is not None:
        cov_post[0] = cov0

    def forecast_single(mu):
        return forecast_dyn(mu[None, :])[0]

    def obs_single(mu):
        return obs_map(mu[None, :])[0]

    def jac_dyn_single(mu):
        return Jacobian_dyn(mu[None, :])[0]

    def jac_obs_single(mu):
        return Jacobian_obs(mu[None, :])[0]

    for nt in range(Nt):
        print(f"EKF step {nt + 1}")

        # 1) Forecast
        mu = mu_post[nt]
        cov = cov_post[nt]
        for _ in range(Nt_sim):
            if Jacobian_dyn is not None:
                F = jac_dyn_single(mu)
            else:
                F = torch.autograd.functional.jacobian( forecast_single, mu.detach().requires_grad_(True) )
            mu = mu + forecast_single(mu) * dt_sim
            # cov = cov + (F @ cov + cov @ F.T + cov_dyn) * dt_sim
            # cov = 0.5 * (cov + cov.T) + 1e-9 * I
            A = torch.matrix_exp(F * dt_sim)
            cov = A @ cov @ A.T + cov_dyn * dt_sim
            cov = 0.5 * (cov + cov.T) + 1e-7 * I
        if Jacobian_obs is not None:
            H = jac_obs_single(mu)
        else:
            H = torch.autograd.functional.jacobian( obs_single, mu.detach().requires_grad_(True) )
        mu_y = obs_single(mu)

        # 2) Analysis
        cov_yy = H @ cov @ H.T + cov_obs
        cov_yy = cov_yy + 1e-9 * torch.eye(obs_size, device=device, dtype=dtype)
        cov_xy = cov @ H.T
        K = torch.linalg.solve(cov_yy.T, cov_xy.T).T
        mu_post[nt + 1] = mu + K @ (y[nt] - mu_y)
        KH = K @ H
        cov = (I - KH) @ cov @ (I - KH).T + K @ cov_obs @ K.T
        cov = 0.5 * (cov + cov.T) + 1e-9 * I
        cov_post[nt + 1] = cov
    return mu_post[1:], cov_post[1:]

def UKF(y, state_dim, forecast_dyn, obs_map, cov_dyn, cov_obs, dt_obs, dt_sim, alpha, k, beta=2, mu0=None, cov0=None):

    def moment2sigma(mu, cov):
        sigma_pts = torch.zeros(2*state_dim + 1, state_dim, device=device)
        S = torch.linalg.cholesky(c*cov)
        sigma_pts[0] = mu
        for i in range(state_dim):
            sigma_pts[i+1] = mu + S[:, i]
            sigma_pts[i+state_dim+1] = mu - S[:, i]
        return sigma_pts

    def sigma2moment(sigma_pts):
        mu = w_mean @ sigma_pts
        cov = ((sigma_pts - mu).T * w_cov) @ (sigma_pts - mu)
        return mu, cov

    device = y.device
    Nt, obs_size = y.shape
    Nt_sim = int(dt_obs / dt_sim)
    I = torch.eye(state_dim, device=device)
    lam = alpha**2 * (state_dim+k) - state_dim
    c = state_dim + lam
    w_mean = 1/(2*c) * torch.ones(2*state_dim+1 , device=device)
    w_cov = 1/(2*c) * torch.ones(2*state_dim+1 , device=device)
    w_mean[0] = lam/c
    w_cov[0] = lam/c + (1 - alpha**2 + beta)
    mu_post = torch.zeros(Nt+1, state_dim, device=device)
    cov_post = I.unsqueeze(0).repeat(Nt+1, 1, 1)
    if mu0 is not None:
        mu_post[0] = mu0
    if cov0 is not None:
        cov_post[0] = cov0
    for nt in range(Nt):
        print(rf"UKF step {nt+1}")
        # 1) Forecast (sigma points transition)
        sigma_pts = moment2sigma(mu_post[nt], cov_post[nt])
        for _ in range(Nt_sim):
            sigma_pts = sigma_pts + forecast_dyn(sigma_pts)*dt_sim
            mu, cov = sigma2moment(sigma_pts)
            cov = cov + cov_dyn*dt_sim
            sigma_pts = moment2sigma(mu, cov)
        gamma_pts = obs_map(sigma_pts)
        # 2) Analysis
        mu_prior, cov_prior = sigma2moment(sigma_pts)
        mu_y = w_mean @ gamma_pts
        cov_yy = ((gamma_pts - mu_y).T * w_cov) @ (gamma_pts - mu_y) + cov_obs
        ccov_xy = ((sigma_pts - mu_prior).T * w_cov) @ (gamma_pts - mu_y)
        K = torch.linalg.solve(cov_yy.T, ccov_xy.T).T
        mu_post[nt+1] = mu_prior + K@(y[nt] - mu_y)
        cov_post[nt+1] = cov_prior - K@ccov_xy.T
    return mu_post[1:], cov_post[1:]

def EnKF(y, state_dim, forecast, obs_map, cov_obs, J=100, x0=None, inflation=None, process_noise_coef=None, GC_xy=None):
    def cross_cov(X, Y):
        n = X.shape[0]
        assert n == Y.shape[0]
        X_centered = X - torch.mean(X, dim=0)
        Y_centered = Y - torch.mean(Y, dim=0)
        cross_cov_matrix = X_centered.T @ Y_centered / (n - 1)
        return cross_cov_matrix

    Nt, dim_y = y.shape
    obs_nosie_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(dim_y, device=y.device), covariance_matrix=cov_obs.to(y.device))

    x_ens = torch.randn(J, Nt+1, state_dim).to(y.device)
    if x0 is not None:
        x_ens[:, 0] = x0
    for nt in range(Nt):
        print(rf"EnKF step {nt+1}")
        # Forecast
        x_ens_prior = forecast(x_ens[:, nt])
        if inflation is not None:
            x_mean = x_ens_prior.mean(dim=0, keepdim=True)
            x_ens_prior = x_mean + inflation * (x_ens_prior - x_mean)
        if process_noise_coef is not None:
            x_ens_prior = x_ens_prior + process_noise_coef * torch.randn_like(x_ens_prior)
        h_ens_prior = obs_map(x_ens_prior)
        # Analysis
        cov_hh = torch.cov(h_ens_prior.T)
        ccov_xy = cross_cov(x_ens_prior, h_ens_prior)
        if GC_xy is not None:
            ccov_xy = ccov_xy * GC_xy
        K = ccov_xy @ torch.inverse(cov_hh + cov_obs)
        # K = torch.linalg.solve((cov_hh + cov_obs).T, ccov_xy.T).T
        y_perturb = y[nt] + obs_nosie_dist.sample((J,))
        x_ens[:, nt+1] = x_ens_prior + (y_perturb - h_ens_prior)@K.T
    x_ens = x_ens[:, 1:]
    return x_ens

def PF(y, state_dim, forecast, obs_map, cov_obs, J=100, x0=None, ess_frac=0.5, process_noise_coef=None, roughening=None):
    # Bootstrap Particle Filter with Adaptive Resampling

    # y: (N, obs_dim)
    device = y.device
    Nt, obs_size = y.shape
    obs_noise_dist = torch.distributions.MultivariateNormal(loc=torch.zeros(obs_size, device=device), covariance_matrix=cov_obs.to(device=device))

    x_particles_history = torch.randn(J, Nt+1, state_dim, device=device) # (J, T, S)
    posterior_mean_history = torch.zeros(Nt, state_dim, device=device) # (J, T, S)
    posterior_var_history = torch.zeros(Nt, state_dim, device=device) # (J, T, S)
    if x0 is not None:
        x_particles_history[:, 0] = x0
    logw = -torch.log(J*torch.ones(J, device=device))
    for nt in range(Nt):
        print(rf"PF step {nt+1}")
        # 1) Forecast
        x_particles_prior = forecast(x_particles_history[:, nt])  # (J, state_dim)
        if process_noise_coef is not None:
            x_particles_prior = x_particles_prior + process_noise_coef * torch.randn_like(x_particles_prior)
        h_particles_prior = obs_map(x_particles_prior) # (J, obs_size)
        # 2) Analysis
        log_likelihood_particles = obs_noise_dist.log_prob(y[nt] - h_particles_prior)  # (J,)
        logw = logw + log_likelihood_particles # update
        logw = logw - torch.logsumexp(logw, dim=0) # normalize
        w = torch.exp(logw)
        ess = 1.0 / torch.sum(w * w)
        posterior_mean = torch.sum(w[:, None] * x_particles_prior, dim=0)
        posterior_var = torch.sum(w[:, None] * (x_particles_prior - posterior_mean[None, :]) ** 2, dim=0)
        posterior_std = torch.sqrt(posterior_var)
        posterior_mean_history[nt] = posterior_mean
        posterior_var_history[nt] = posterior_var
        if ess < (ess_frac * J):
            # resampling
            indices = torch.multinomial(w, num_samples=J, replacement=True)
            x_particles_history[:, nt + 1] = x_particles_prior[indices]
            logw = -torch.log(J*torch.ones(J, device=device))
            if roughening is not None and roughening > 0:
                x_particles_history[:, nt+1] += roughening * posterior_std[None, :] * torch.randn_like(x_particles_history[:, nt+1])
        else:
            #  no resampling (keep forecast particles)
            x_particles_history[:, nt + 1] = x_particles_prior
    return posterior_mean_history, posterior_var_history

