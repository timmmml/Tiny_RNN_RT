"""Customized rnn layers.

Mainly GRU, MIGRU, SGRU, PNR, etc.
In nn.Module way.
"""
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import numbers
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    def forward(self, input, state):
        hx = state[0, ...] # ignore first dim, seq_len=1
        gates_i = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gates_h = torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_i + reset_h)
        update_gate = torch.sigmoid(update_i + update_h)
        new_gate = torch.tanh(new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy[None, ...] # insert back first dim

class LRCell(nn.Module):
    def __init__(self, input_size, hidden_size, rank, nonlinearity='tanh'):
        super(LRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh_m = Parameter(torch.randn(hidden_size, rank))
        self.weight_hh_n = Parameter(torch.randn(rank, hidden_size))
        self.bias = Parameter(torch.randn(hidden_size))
        self.nonlinearity = nonlinearity

    def forward(self, input, state):
        hx = state[0, ...] # ignore first dim, seq_len=1
        weight_hh = torch.mm(self.weight_hh_m, self.weight_hh_n)
        hy = torch.mm(input, self.weight_ih.t()) + torch.mm(hx, weight_hh.t()) + self.bias
        if self.nonlinearity == 'tanh':
            hy = torch.tanh(hy)
        elif self.nonlinearity == 'relu':
            hy = torch.relu(hy)
        else:
            raise NotImplementedError
        return hy[None, ...] # insert back first dim

class MIGRUCell(nn.Module):
    """multiplicative integration GRU"""
    def __init__(self, input_size, hidden_size):
        super(MIGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        r2 = math.sqrt(1/hidden_size)
        r1 = - r2
        self.weight_ih = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size, input_size)))
        self.weight_hh = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size, hidden_size)))
        self.bias_ih = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size,))/100) # close to 0
        self.bias_hh = Parameter(self.uniform_parameter(r1, r2, (3 * hidden_size,))/100)
        self.alpha_ih = Parameter(torch.zeros(3 * hidden_size)) # randn
        self.beta_i = Parameter(torch.ones(3 * hidden_size)) # randn
        self.beta_h = Parameter(torch.ones(3 * hidden_size)) # randn

    def uniform_parameter(self, r1, r2, size):
        temp = (r2 - r1) * torch.rand(*size) + r1
        return temp

    def forward(self, input, state):
        hx = state[0] # ignore first dim, seq_len=1
        input_cur = torch.mm(input, self.weight_ih.t())
        hx_cur = torch.mm(hx, self.weight_hh.t())
        gates_i = self.beta_i * input_cur + self.bias_ih
        gates_h = self.beta_h * hx_cur + self.bias_hh
        gates_ih = self.alpha_ih * input_cur * hx_cur

        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_ih, update_ih, new_ih = gates_ih.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_ih + reset_i + reset_h)
        update_gate = torch.sigmoid(update_ih + update_i + update_h)
        new_gate = torch.tanh(new_ih * reset_gate + new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy[None, ...] # insert back first dim

class SGRUCell(nn.Module):
    """switching GRU.

    Modification of the original GRU. Because the input is one-hot (for combinations of discrete input features),
    will use different recurrent weight depending on the input.
    """
    def __init__(self, input_size, hidden_size):
        super(SGRUCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot encoding
        self.hidden_size = hidden_size
        #self.weight_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # TODO: use this for continous input dimension
        self.weight_hh = Parameter(torch.randn(3 * hidden_size, hidden_size, input_size)) # 3H*H*I
        self.bias_ih = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I
        self.bias_hh = Parameter(torch.randn(3 * hidden_size, input_size)) # 3H*I

    def forward(self, input, state):
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        hx = state[0] # B*H # ignore first dim, seq_len=1
        trial_weight_hh = (self.weight_hh[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*3H*H*I, B*1*1*I-> B*3H*H
        rec_temp = (trial_weight_hh * hx[:, None, :]).sum(-1) # B*3H*H, B*1*H->B*3H
        trial_bias_ih = (self.bias_ih[None, :,:] * input[:, None, :]).sum(-1) # 1*3H*I, B*1*I-> B*3H
        trial_bias_hh = (self.bias_hh[None, :,:] * input[:, None, :]).sum(-1) # 1*3H*I, B*1*I-> B*3H
        gates_i = trial_bias_ih
        gates_h = rec_temp + trial_bias_hh
        reset_i, update_i, new_i = gates_i.chunk(3, 1)
        reset_h, update_h, new_h = gates_h.chunk(3, 1)
        reset_gate = torch.sigmoid(reset_i + reset_h)
        update_gate = torch.sigmoid(update_i + update_h)
        new_gate = torch.tanh(new_i + reset_gate * new_h)
        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy[None, ...] # insert back first dim


class PNRCell(nn.Module):
    """Polynomial regression model.
    Each time step is a polynomial transfomation.
    Only support hidden_size<=2 and order<=3 for now.

    Attributes:
        po: polynomial_order
    """
    def __init__(self, input_size, hidden_size, polynomial_order=0):
        super(PNRCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot
        self.hidden_size = hidden_size
        self.po = polynomial_order
        assert polynomial_order>0, "polynomial_order not provided"
        feature_size = 0
        if self.hidden_size == 1:
            feature_size += self.po
        elif self.hidden_size == 2:
            if self.po >= 1:
                feature_size += 2
            if self.po >= 2:
                feature_size += 3
            if self.po >= 3:
                feature_size += 4
            if self.po >= 4:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.feature_size = feature_size
        self.weight = Parameter(torch.zeros(hidden_size, feature_size, input_size)) # H*F*I
        self.bias = Parameter(torch.zeros(hidden_size, input_size)) # H*I


    def forward(self, input, state):
        # input is one-hot: batch_size, input_size
        assert (input.sum(-1) == 1).all()
        trial_weight = (self.weight[None, :,:,:] * input[:, None, None, :]).sum(-1) # 1*H*F*I, B*1*1*I-> B*H*F
        trial_bias = (self.bias[None, :,:] * input[:, None, :]).sum(-1) # 1*H*I, B*1*I-> B*H
        hx = state[0] # B*H # ignore first dim, seq_len=1
        batch_size = hx.shape[0]
        features = []
        if self.hidden_size == 1:
            h1 = hx[:, 0]
            if self.po >= 1:
                features += [h1]
            if self.po >= 2:
                features += [h1**2]
            if self.po >= 3:
                features += [h1**3]
            if self.po >= 4:
                raise NotImplementedError
        elif self.hidden_size == 2:
            h1 = hx[:, 0]
            h2 = hx[:, 1]
            if self.po >= 1:
                features += [h1, h2]
            if self.po >= 2:
                features += [h1**2, h2**2, h1*h2]
            if self.po >= 3:
                features += [h1**3, h1**2*h2, h1*h2**2,h2**3]
            if self.po >= 4:
                raise NotImplementedError
        elif self.hidden_size == 3:
            h1 = hx[:, 0]
            h2 = hx[:, 1]
            h3 = hx[:, 2]
            if self.po >= 1:
                features += [h1, h2, h3]
            if self.po >= 2:
                features += [h1**2, h2**2, h3**2, h1*h2, h2*h3, h1*h3]
            if self.po >= 3:
                raise NotImplementedError
        else:
            raise NotImplementedError
        features = torch.stack(features, dim=1) # B, F
        rec_temp = (trial_weight * features[:, None, :]).sum(-1) # B*H*F, B*1*F->B*H
        hy = hx + rec_temp + trial_bias

        return hy[None, ...] # insert back first dim


class PNRCellSymm(PNRCell):
    """Polynomial regression model.
    Each time step is a polynomial transfomation.
    Only support hidden_size<=2 and order<=3 for now.

    Attributes:
        po: polynomial_order
    """
    def __init__(self, input_size, hidden_size, polynomial_order=0):
        super(PNRCell, self).__init__()
        self.input_size = input_size # input_size is for one-hot
        self.hidden_size = hidden_size
        self.po = polynomial_order
        assert polynomial_order>0, "polynomial_order not provided"
        feature_size = 0
        if self.hidden_size == 1:
            feature_size += self.po
        elif self.hidden_size == 2:
            if self.po >= 1:
                feature_size += 2
            if self.po >= 2:
                feature_size += 3
            if self.po >= 3:
                feature_size += 4
            if self.po >= 4:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.feature_size = feature_size
        self._weight = Parameter(torch.zeros(hidden_size, feature_size, input_size)) # H*F*I
        self.bias = Parameter(torch.zeros(hidden_size, input_size)) # H*I


    @property
    def weight(self):
        return (self._weight + self._weight.transpose(0, 1)) / 2

class RNNLayer_custom(nn.Module):
    """Customized RNN layer.

    Attributes:
        rnn_type:
        rnncell: a cell responsible for a single time step.
    """
    def __init__(self, *cell_args, **kwargs):
        super().__init__()
        self.rnn_type = kwargs['rnn_type']
        if self.rnn_type == 'SGRU':
            self.rnncell = SGRUCell(*cell_args)
        elif self.rnn_type == 'MIGRU':
            self.rnncell = MIGRUCell(*cell_args)
        elif self.rnn_type == 'GRU':
            self.rnncell = GRUCell(*cell_args)
        elif 'PNR' in self.rnn_type:
            if 'symm' in kwargs and kwargs['symm']:
                self.rnncell = PNRCellSymm(*cell_args, polynomial_order=kwargs['polynomial_order'])
            else:
                self.rnncell = PNRCell(*cell_args, polynomial_order=kwargs['polynomial_order'])
        elif 'LR' in self.rnn_type:
            if 'nonlinearity' in kwargs:
                nonlinearity = kwargs['nonlinearity']
            else:
                nonlinearity = 'tanh'
            self.rnncell = LRCell(*cell_args, rank=kwargs['rank'], nonlinearity=nonlinearity)
        else:
            print(self.rnn_type)
            raise NotImplementedError

    def forward(self, input, state):
        """Run a RNN cell for several time steps.

        Args:
            input: shape: seq_len, batch_size, input_dim
            state: shape: seq_len=1, batch_size, hidden_dim

        Returns:
            outputs: shape: seq_len, batch_size, output_dim
            state: final state after seeing all inputs, shape: seq_len=1, batch_size, hidden_dim
        """
        assert len(state.shape) == 3
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            state = self.rnncell(inputs[i], state)
            out = state
            outputs += [out]
        assert len(state.shape) == 3
        return torch.cat(outputs, 0), state


def test_script_pnr_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(1,batch, hidden_size)
    rnn = PNRCell(input_size, hidden_size, polynomial_order=3)
    out = rnn(inp[0], state)
    num_params = sum(param.numel() for param in rnn.parameters())
    print('Net num_params', num_params)
    print(state, out, )

def test_script_migru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size).float()
    state = torch.randn(1, batch, hidden_size)
    rnn = MIGRUCell(input_size, hidden_size)
    out  = rnn(inp[0], state)
    #print(out, )


def test_script_gruswitch_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randint(0,input_size,(seq_len, batch))
    inp = nn.functional.one_hot(inp, num_classes=input_size)
    state = torch.randn(1, batch, hidden_size)
    rnn = SGRUCell(input_size, hidden_size)
    out  = rnn(inp[0], state)
    #print(out, )

def test_script_gru_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = torch.randn(1, batch, hidden_size)
    rnn = RNNLayer_custom(input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    gru = nn.GRU(input_size, hidden_size, 1)
    gru_state = state.unsqueeze(0)
    for gru_param, custom_param in zip(gru.all_weights[0], rnn.parameters()):
        assert gru_param.shape == custom_param.shape
        with torch.no_grad():
            gru_param.copy_(custom_param)
    gru_out, gru_out_state = gru(inp, gru_state)
    assert (out - gru_out).abs().max() < 1e-5, out
    assert (out_state - gru_out_state).abs().max() < 1e-5, out_state


if __name__ == '__main__':
    test_script_pnr_layer(5, 2, 4, 2)
    test_script_migru_layer(5, 2, 4, 3)
    test_script_gruswitch_layer(5, 2, 4, 3)

