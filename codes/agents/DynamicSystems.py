import pandas as pd
import torch
from scipy.linalg import expm
from path_settings import *
from numba import jit
import numpy as np
from torch import jit as jit_torch
from functools import partial
import torch.nn as nn

class ISNNet(nn.Module):
    def __init__(self, config):
        super(ISNNet, self).__init__()
        self.network = pd.read_csv(
            str(OBJECT_PATH) + "/network_dynamics/soc_lognormal_big.txt", sep="\t", header=None
        ).to_numpy()[:, :-1]
        torch.set_default_dtype(torch.float32)
        self.W = self.network
        self.tau = 500 # time constant is 500 ms, such that the network would be be able to retain something
        self.dt = None
        if hasattr(config, "dt"):
            self.set_dt(config.dt)

    def set_dt(self, dt): 
        """NOTE: dt is the time step measured in time units. First need to convert your N steps per trial into this, 
        by taking the reciprocal and considering trial time constant"""
        self.eW = expm((self.network - np.eye(self.network.shape[0])) * dt/self.tau)
        self.W_inv_eW = np.linalg.inv(self.network) @ (self.eW - np.eye(self.network.shape[0]))
        self.eW_torch = nn.Parameter(torch.tensor(self.eW))

        self.W_inv_eW_torch = nn.Parameter(torch.tensor(self.W_inv_eW))
        self.dt = dt
        self.dynamics_with_input_torch = partial(jit_torch.script(dynamics_with_input_torch_), eW=torch.tensor(self.eW), W_inv_eW=torch.tensor(self.W_inv_eW))
        self.dynamics_with_input_np = partial(dynamics_with_input_np_, eW = self.eW, W_inv_eW = self.W_inv_eW)
    
    def set_trainable(self, trainable=True):
        self.eW_torch.requires_grad = trainable
        self.W_inv_eW_torch.requires_grad = trainable
    
    def dynamics_with_inputs(self, x0, I): 
        x = torch.zeros((x0.shape[0], I.shape[-2], x0.shape[-1]), dtype=torch.float32, device = x0.device)
        x0 = x0.float()
        x[:, 0, :] = x0
        I = I.float()
        eW = self.eW_torch.float().to(x0.device)
        W_inv_eW = self.W_inv_eW_torch.float().to(x0.device)
        for i in range(1, I.shape[-2]):
            x[:, i, :] = (
                (eW @ x[:, i - 1, :].T).T + (W_inv_eW @ I[:, i - 1, :].T).T
            )
        return x
    
    # def to(self, device):
    #     self.
    #     self.eW_torch.to(device)
    #     self.W_inv_eW_torch.to(device)
    #     return self


@jit(nopython=True)
def dynamics_with_input_np_(x0, I, eW, W_inv_eW):
    """This method computes the dynamics of the network given the initial condition x and the input dynamics.

    Args:
        x0 (np.array(batch_size, self.state_dim)): the initial condition tensor
        I (np.array(batch_size, full_seq_length (input, discretised into the system time already and padded with zeros if necessary and unbiased), self.state_dim)): the input transformed by state dim already

    Returns:
        np.array(batch_size, self.state_dim): the final state of the network after the dynamics
    """
    x = np.zeros((x0.shape[0], I.shape[-2], x0.shape[-1]), dtype=np.float32)
    x0 = x0.astype(np.float32)
    x[:, 0, :] = x0
    I = I.astype(np.float32)
    eW = eW.astype(np.float32)
    W_inv_eW = W_inv_eW.astype(np.float32)
    for i in range(1, I.shape[-2]):
        x[:, i, :] = (
            (eW @ x[:, i - 1, :].T).T + (W_inv_eW @ I[:, i - 1, :].T).T
        )
    return x

def dynamics_with_input_torch_(x0, I, eW, W_inv_eW):
    x = torch.zeros((x0.shape[0], I.shape[-2], x0.shape[-1]), dtype=torch.float32, device = x0.device)
    x0 = x0.float()
    x[:, 0, :] = x0
    I = I.float()
    eW = eW.float().to(x0.device)
    W_inv_eW = W_inv_eW.float().to(x0.device)
    for i in range(1, I.shape[-2]):
        x[:, i, :] = (
            (eW @ x[:, i - 1, :].T).T + (W_inv_eW @ I[:, i - 1, :].T).T
        )
    return x



