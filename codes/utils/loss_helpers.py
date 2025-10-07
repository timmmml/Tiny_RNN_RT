import torch

def interpolate_exp(N):
    return torch.exp(torch.arange(0, N, step = 10/N) - 10)/N

def interpolate_exp_signed(N):
    N_2 = N//2
    signs = torch.cat([-torch.ones(N_2), torch.ones(N - N_2)])
    return torch.exp(torch.arange(0, N, step = 10/N) - 10)/N * signs

def interpolate_zeros(N): 
    z = torch.zeros(N)
    z[-1] = 1
    return z