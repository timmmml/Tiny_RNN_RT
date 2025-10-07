"""A collection of implemented recurrent networks.

Mainly RNN net and LSTM net.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import os
import gc
import platform
FORCE_USING_TORCH_SCRIPT = False

if FORCE_USING_TORCH_SCRIPT is True or (FORCE_USING_TORCH_SCRIPT is None and platform.system() == 'Windows'):
    from .custom_lstms import RNNLayer_custom
    # torchscript can save ~25% time for the current layers
    # cannot work together with multiprocessing, can be used for single process training
    # current code will trigger an internal bug in pytorch on Linux
else:
    from .custom_rnn_layers import RNNLayer_custom

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class RNNnet(nn.Module):
    """ A RNN network: input layer + recurrent layer + a readout layer.

    Attributes:
        input_dim:
        hidden_dim:
        output_dim:
        output_h0: whether the output of the network should contain the initial networks' hidden state
        rnn: the recurrent layer
        h0: the initial networks' hidden state
        readout_FC: whether the readout layer is full connected or not
        lin: the full connected readout layer
        lin_coef: the inverse temperature of a direct readout layer
    """
    def __init__(self, input_dim, hidden_dim, output_dim, readout_FC=True, trainable_h0=False, rnn_type='GRU', output_h0='False',**kwargs):
        """
        Args:
            input_dim:
            hidden_dim:
            output_dim:
            output_h0:
            readout_FC:
            rnn_type:
            trainable_h0: the agent's initial hidden state trainable or not
        """
        super(RNNnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_h0 = output_h0
        self.include_embedding = False
        if 'include_embedding' in kwargs and kwargs['include_embedding']:
            self.num_embeddings = kwargs['num_embeddings'] # the number of subjects
            self.embedding_dim = kwargs['embedding_dim'] # the dimension of the embedding
            zero_embedding = torch.zeros(self.num_embeddings, self.embedding_dim)
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim, _weight=zero_embedding)
            self.include_embedding = True
            input_dim += self.embedding_dim
        if rnn_type == 'GRU': # official GRU implementation
            self.rnn = nn.GRU(input_dim, hidden_dim)
        else: # customized RNN layers in TorchScript
            self.rnn = RNNLayer_custom(input_dim, hidden_dim, rnn_type=rnn_type, **kwargs)
        self.readout_FC = readout_FC
        if 'complex_readout' in kwargs:
            self.complex_readout = kwargs['complex_readout']
            if self.complex_readout:
                output_dim *= 2 # self.output_dim is not changed
        else:
            self.complex_readout = False
        if readout_FC:
            self.lin = nn.Linear(hidden_dim, output_dim)
        else:
            assert hidden_dim == output_dim, (hidden_dim, output_dim)
            self.lin_coef = nn.Parameter(torch.ones(1,1,1))
        if trainable_h0:
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.h0 = torch.zeros(1, 1, hidden_dim).double()
        self.dummy_param = nn.Parameter(torch.empty(0)) # a dummy parameter to store the device of the model

    def forward(self, input, get_rnnout=False, h0=None):
        """
        Args:
            input: shape: seq_len, batch_size, input_size
            get_rnnout: whether the internal states of rnn should be outputted
            h0: whether use a customized h0 or default h0
                shape: seq_len=1, batch_size=1, hidden_dim

        Returns:
            Return the final output of the RNN
            Also return the internal states if get_rnnout=True
        """
        model_device = self.dummy_param.device
        if self.h0.device != model_device: # move h0 to the same device as the model if h0 is not Parameter
            self.h0 = self.h0.to(model_device) # this should be done before the first forward pass
        if h0 is None:
            h0 = self.h0
        assert input.device == model_device, (input.device, model_device)
        assert h0.device == model_device, (h0.device, model_device)
        seq_len, batch_size, input_dim = input.shape
        h0_expand = h0.repeat(1, batch_size, 1) # h0 is the same for each sample in the batch
        if self.include_embedding:
            input, embedding_input = input[..., :-1], input[..., -1]
            assert input.shape[-1] == self.input_dim, (input.shape, self.embedding_dim)
            embedding_input = embedding_input.long()
            embedding = self.embedding(embedding_input)
            assert embedding.shape == (seq_len, batch_size, self.embedding_dim), (embedding.shape, (seq_len, batch_size, self.embedding_dim))
            input = torch.cat((input, embedding), -1)
        rnn_out, hn = self.rnn(input, h0_expand)  # rnn_out shape: seq_len, batch, hidden_size
        if self.output_h0:
            rnn_out = torch.cat((h0_expand, rnn_out), 0)
            seq_len += 1
        if self.readout_FC:
            scores = self.lin(rnn_out.view(seq_len * batch_size, self.hidden_dim))
        else:
            scores = self.lin_coef * rnn_out
        if hasattr(self, 'complex_readout') and self.complex_readout:
            scores = scores.view(seq_len, batch_size, self.output_dim*2)
            scores_real = scores[..., :self.output_dim]
            scores_imag = scores[..., self.output_dim:]
            scores = scores_real ** 2 + scores_imag ** 2
            scores = torch.log(scores + 1e-8) # pseudo logit for complex numbers
        else:
            scores = scores.view(seq_len, batch_size, self.output_dim)
        if get_rnnout:
            return scores, rnn_out
        return scores

class LSTMnet(nn.Module):
    """ A LSTM network: input layer + recurrent layer + a readout layer.
    TODO: this class should have the same API as RNNnet
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, get_rnnout=False):
        seq_len, batch, _ = input.shape
        rnn_out, (hn, cn) = self.lstm(input) # seq_len, batch, hidden_size
        scores = self.lin(rnn_out.view(seq_len * batch, self.hidden_dim))
        if get_rnnout:
            return scores.view(seq_len, batch, self.output_dim),rnn_out
        return scores.view(seq_len, batch, self.output_dim)

class VRNNnet(nn.Module):
    """ A vanilla RNN network: input layer + recurrent layer + a readout layer.
    TODO: this class should be incorporated into RNNnet.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='relu'):
        super(VRNNnet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, nonlinearity=nonlinearity)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, get_rnnout=False):
        seq_len, batch, _ = input.shape
        rnn_out, hn = self.rnn(input) # seq_len, batch, hidden_size
        scores = self.lin(rnn_out.view(seq_len * batch, self.hidden_dim))
        if get_rnnout:
            return scores.view(seq_len, batch, self.output_dim),rnn_out
        return scores.view(seq_len, batch, self.output_dim)