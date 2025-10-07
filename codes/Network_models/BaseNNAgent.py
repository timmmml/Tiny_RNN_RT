'''Implements a base class for all neural network agents used in the present project.

Stage 0: feasibility check
    saved_model used: simple RNN structures (specify within config model type (RNN, GRU, LSTM) and hyperparams).


Stage 1 onwards...

'''
import torch
import torch.nn as nn

class BaseNNAgent(nn.Module):
    def __init__(self):
        super(BaseNNAgent, self).__init__()
        self.out = None
        self.dt = 1.0 # Time step

    def forward(self, x):
        raise NotImplementedError
