import torch
import torch.nn.functional as F

def RMSE(x, mu):
    # x: (N, S),  mu: (N, S)
    rmse = F.mse_loss(x, mu).item()**0.5
    return rmse

def NLL(x, mu, cov):
    # x: (N, S), mu: (N, S), cov: (N, S) or (N, S, S)
    state_dim = x.shape[-1]
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    if cov.dim() == 2:
        nll = 0.5*(((x-mu)**2/cov + torch.log(cov)).sum(dim=-1) + state_dim*torch.log(2*pi))
    else:
        x = x.unsqueeze(-1)
        mu = mu.unsqueeze(-1)
        wmse = ((x-mu).mT @ torch.inverse(cov) @ (x-mu)).squeeze((-1, -2))
        nll = 0.5*(wmse + torch.logdet(cov) + state_dim*torch.log(2*pi))
    return nll.mean().item()

def ECP(x, mu, cov, eps=1e-6):
    # x:   (N, S)
    # mu:  (N, S)
    # cov: (N, S) or (N, S, S)
    if cov.dim() == 2:
        var = cov
    else:
        cov = 0.5 * (cov + cov.mT)      # enforce symmetry
        torch.diagonal(cov, dim1=-2, dim2=-1).clamp_(min=0.)
        var = torch.diagonal(cov, dim1=-2, dim2=-1) + eps
    z = 1.96  # 95% Gaussian interval
    std = torch.sqrt(var)
    lower = mu - z * std
    upper = mu + z * std
    covered = (x >= lower) & (x <= upper)
    ecp = covered.float().mean()
    return ecp.item()

def CRPS(x, mu, cov, eps=1e-6):
    # x:   (N, S)
    # mu:  (N, S)
    # cov: (N, S) or (N, S, S)
    if cov.dim() == 2:
        cov = F.softplus(cov) + eps
        var = cov
    else:
        cov = 0.5 * (cov + cov.mT)      # enforce symmetry
        torch.diagonal(cov, dim1=-2, dim2=-1).clamp_(min=0.)
        var = torch.diagonal(cov, dim1=-2, dim2=-1) + eps
    std = torch.sqrt(var)
    z = (x - mu) / std
    zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    normal = torch.distributions.Normal(zero, one)
    pdf = torch.exp(normal.log_prob(z))
    cdf = normal.cdf(z)
    pi = torch.tensor(torch.pi, device=x.device, dtype=x.dtype)
    crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))
    return crps.mean().item()

