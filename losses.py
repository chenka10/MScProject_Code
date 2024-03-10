import torch

def kl_criterion_normal(mu, logvar):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return KLD

def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld