"""
The goal here is to step out of the 1-step agent and benchmark models against more complex agents that are still task-optimized

- This agent is a mapper between the task and a reservoir of underlying dynamics - random, nonlinear but stable

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from path_settings import *
from numba import jit
from scipy.linalg import solve_continuous_lyapunov, expm

import importlib

class DynamicAgent(nn.Module):
    def __init__(self, config): 
        super(DynamicAgent, self).__init__()
        self.config = config
        self.net_source = importlib.import_module(config.get("net_source", "agents.DynamicSystems"))
        self.net_type = config.get("net_type", "ISNNet")
        self.net = getattr(self.net_source, self.net_type)(config.get("net_config", {}))
        self.trial_length = config.get("trial_length", 1000)
        self.N = config.get("N", 10)
        self.dt = self.trial_length / self.N
        self.net.set_dt(self.dt)
        self.net.set_trainable(False)
        
        self.out_dim = config.get("out_dim", 1)
        self.in_dim = config.get("in_dim", 3)
        self.B = nn.Sequential(
            nn.Linear(self.in_dim, 8, bias = False), 
            nn.ReLU(),
            nn.Linear(8, self.net.W.shape[0], bias = False)
        )

        # freeze B
        # for param in self.B.parameters():
            # param.requires_grad = False
        self.C = nn.Sequential(
            nn.Linear(self.net.W.shape[0], self.out_dim, bias = False), 
            nn.Sigmoid()
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def copy_C(self):
        self.C_prime = nn.Linear(self.net.W.shape[0], self.out_dim, bias = False)
        # inherit C's parameters:
        self.C_prime.weight = self.C[0].weight

    def forward(self, x):
        """X comes in as batch_size, seq_len, in_dim"""
        x = x.unsqueeze(-2)
        x = torch.cat([x, torch.zeros_like(x).repeat(1, 1, self.N-1, 1)], dim=-2)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        I = self.B(x)
        I_norm = torch.norm(I, dim=-1, keepdim=True)
        y = self.net.dynamics_with_input_torch(I)
        # now we have batch_size x seq_len*N x W_dim
        # let's shape it to batch_size x seq_len x N*W_dim
        y = y.reshape(x.shape[0], -1, self.net.W.shape[0])
        out = self.C(y)
        return out, I_norm
    
    def simulate(self, task, n_blocks, n_trials, return_x = False, rand_agent = False, reset = True, return_raw_out = False, noise_sd = 0.01):
        # vectorise across blocks, dynamic across trials
        if reset:
            task.reset(n_blocks, n_trials)
        else: 
            task.reset_without_regen()
        x0 = torch.zeros(n_blocks, self.net.W.shape[0], device=self.device)
        I = torch.zeros(n_blocks, self.N+1, self.in_dim, device=self.device)
        # start with random init states for each block; no input
        I_ = self.B(I)
        x = self.net.dynamics_with_input_torch(x0, I_)
        y = self.C(x)

        c = torch.zeros((n_blocks, 0, 1), device=self.device)
        I_norm = 0
        I_norm = sum(sum(I.norm(dim=-1, keepdim=False)))
        out_tot = torch.zeros((n_blocks, 0), device = self.device)
        y_raw = torch.zeros((n_blocks, 0, self.net.W.shape[0]), device = self.device)
        if return_raw_out:
            self.copy_C()
            y_raw = self.C_prime(x) 
        for i in range(n_trials-1): 
            if not rand_agent:
                choice = torch.tensor(y[:, -2, 0] > 0.5, dtype=torch.float32, device=self.device)
            else: 
                choice = torch.tensor(torch.rand_like(y[:, -2, 0] > 0.5, dtype = torch.float32, device=self.device))

            s_s, out, correct_choice = task.trial(choice)

            out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
            c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
            I_ = torch.cat([choice.unsqueeze(1), s_s.unsqueeze(1), out.unsqueeze(1)], dim = 1).reshape(n_blocks, 1, -1)
            I_ = (I_ - 0.5)
            # I = torch.cat([I_, torch.zeros_like(I_).repeat(1, self.N, 1)], dim=1)

            # I_ = torch.cat([I_, I_.repeat(1, self.N//2, 1)], dim=1)/(self.N//2)
            I_ = torch.cat([I_, I_.repeat(1, self.N, 1)], dim=1)/(self.N/2)
            I = torch.cat([I[:, :-1, :], I_], dim=1)
            # I = torch.cat([I_, torch.zeros_like(I_)[:, :-1, :]], dim=1)
            I_ = self.B(I_) 
            I_ += torch.randn_like(I_) * noise_sd
            I_norm += sum(sum(I_.norm(dim=-1, keepdim=False)))
            x_ = self.net.dynamics_with_input_torch(x[:, -1, :], I_)
            x = torch.cat([x[:, :-1, :], x_], dim=1)
            y_ = self.C(x_)
            y = torch.cat([y[:, :-1, :], y_], dim=1)
            if return_raw_out:
                y_raw = torch.cat([y_raw[:, :-1, :], self.C_prime(x_)], dim=1)
        I = I[:, :-1, :]
        _, out, correct_choice = task.trial(choice)
        out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
        c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
        if return_x: 
            if return_raw_out:
                return y[:, :-1, :], c, I, x, out_tot, y_raw[:, :-1, :]
            return y[:, :-1, :], c, I, x, out_tot
        if return_raw_out:
            return y[:, :-1, :], c, I, y_raw[:, :-1, :]
        return y[:, :-1, :], c, I_norm

    def to(self, device):
        self.device = device
        self.net.to(device)
        self.B.to(device)
        self.C.to(device)
        return self
 
class LSTMNet(nn.Module): 
    def __init__(self, config) -> None:
        super(LSTMNet, self).__init__()
        self.config = config       
        self.N = config.get("N", 10)
        self.in_dim = config.get("in_dim", 3)
        self.out_dim = config.get("out_dim", 1)
        self.hidden_dim = config.get("hidden_dim", 200)
        self.rnn = nn.LSTM(self.in_dim, self.hidden_dim, batch_first = True)
        self.C = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim, bias = False),
            nn.Sigmoid()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def copy_C(self): 
        self.C_prime = nn.Linear(self.hidden_dim, self.out_dim, bias = False)
        self.C_prime.weight = self.C[0].weight
    
    def simulate(self, task, n_blocks, n_trials, return_x = False, rand_agent = False, reset = True, return_raw_out = False, noise_sd = 0):
        # vectorise across blocks, dynamic across trials
        if reset:
            task.reset(n_blocks, n_trials)
        else: 
            task.reset_without_regen()

        x0 = torch.zeros(n_blocks, self.hidden_dim, device=self.device).unsqueeze(0)
        I = torch.zeros(n_blocks, self.N+1, self.in_dim, device=self.device)
        # start with random init states for each block; no input
        x, (x_n, c_n) = self.rnn.forward(I, (x0, torch.zeros_like(x0)))
        y = self.C(x)

        c = torch.zeros((n_blocks, 0, 1), device=self.device)
        I_norm = 0
        out_tot = torch.zeros((n_blocks, 0), device = self.device)
        y_raw = torch.zeros((n_blocks, 0, self.hidden_dim), device = self.device)
        if return_raw_out:
            self.copy_C()
            y_raw = self.C_prime(x) 
        for i in range(n_trials-1): 
            if not rand_agent:
                choice = torch.tensor(y[:, -2, 0] > 0.5, dtype=torch.float32, device=self.device)
            else: 
                choice = torch.tensor(torch.rand_like(y[:, -2, 0] > 0.5, dtype = torch.float32, device=self.device))

            s_s, out, correct_choice = task.trial(choice)
            out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
            c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
            I_ = torch.cat([choice.unsqueeze(1), s_s.unsqueeze(1), out.unsqueeze(1)], dim = 1).reshape(n_blocks, 1, -1)
            # I = torch.cat([I_, torch.zeros_like(I_).repeat(1, self.N, 1)], dim=1)
            I = torch.cat([I_, I_.repeat(1, self.N, 1)], dim=1)
            I += torch.randn_like(I) * noise_sd
            
            x_, (x_n, c_n) = self.rnn(I, (x_n, c_n))
            x = torch.cat([x[:, :-1, :], x_], dim=1)
            y_ = self.C(x_)
            y = torch.cat([y[:, :-1, :], y_], dim=1)
            if return_raw_out:
                y_raw = torch.cat([y_raw[:, :-1, :], self.C_prime(x_)], dim=1)

        _, out, correct_choice = task.trial(choice)
        out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
        c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
        if return_x: 
            if return_raw_out:
                return y[:, :-1, :], c, I_norm, x, out_tot, y_raw
            return y[:, :-1, :], c, I_norm, x, out_tot
        if return_raw_out:
            return y[:, :-1, :], c, I_norm, y_raw 
        return y[:, :-1, :], c, I_norm
    
    def to(self, device):
        self.device = device
        self.rnn.to(device)
        self.C.to(device)
        return self

   

class GRUNet(nn.Module): 
    def __init__(self, config) -> None:
        super(GRUNet, self).__init__()
        self.config = config       
        self.N = config.get("N", 10)
        self.in_dim = config.get("in_dim", 3)
        self.out_dim = config.get("out_dim", 1)
        self.hidden_dim = config.get("hidden_dim", 200)
        self.rnn = nn.GRU(self.in_dim, self.hidden_dim, batch_first = True)
        self.C = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim, bias = False),
            nn.Sigmoid()
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def copy_C(self): 
        self.C_prime = nn.Linear(self.hidden_dim, self.out_dim, bias = False)
        self.C_prime.weight = self.C[0].weight
    
    def simulate(self, task, n_blocks, n_trials, return_x = False, rand_agent = False, reset = True, return_raw_out = False, noise_sd = 0):
        # vectorise across blocks, dynamic across trials
        if reset:
            task.reset(n_blocks, n_trials)
        else: 
            task.reset_without_regen()

        x0 = torch.randn(n_blocks, self.hidden_dim, device=self.device).unsqueeze(0)
        I = torch.zeros(n_blocks, self.N+1, self.in_dim, device=self.device)
        # start with random init states for each block; no input
        x, (x_n) = self.rnn.forward(I, (x0))
        y = self.C(x)

        c = torch.zeros((n_blocks, 0, 1), device=self.device)
        I_norm = 0
        out_tot = torch.zeros((n_blocks, 0), device = self.device)
        y_raw = torch.zeros((n_blocks, 0, self.hidden_dim), device = self.device)
        if return_raw_out:
            self.copy_C()
            y_raw = self.C_prime(x) 
        for i in range(n_trials-1): 
            if not rand_agent:
                choice = torch.tensor(y[:, -2, 0] > 0.5, dtype=torch.float32, device=self.device)
            else: 
                choice = torch.tensor(torch.rand_like(y[:, -2, 0] > 0.5, dtype = torch.float32, device=self.device))

            s_s, out, correct_choice = task.trial(choice)
            out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
            c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
            I_ = torch.cat([choice.unsqueeze(1), s_s.unsqueeze(1), out.unsqueeze(1)], dim = 1).reshape(n_blocks, 1, -1)
            I = torch.cat([I_, torch.zeros_like(I_).repeat(1, self.N, 1)], dim=1)
            I += torch.randn_like(I) * noise_sd
            
            x_, (x_n) = self.rnn(I, (x_n))
            x = torch.cat([x[:, :-1, :], x_], dim=1)
            y_ = self.C(x_)
            y = torch.cat([y[:, :-1, :], y_], dim=1)
            if return_raw_out:
                y_raw = torch.cat([y_raw[:, :-1, :], self.C_prime(x_)], dim=1)

        _, out, correct_choice = task.trial(choice)
        out_tot = torch.cat((out_tot, out.unsqueeze(1)), dim = -1)
        c = torch.cat([c, correct_choice.unsqueeze(1).unsqueeze(1).repeat(1, self.N, 1)], dim=1)
        if return_x: 
            if return_raw_out:
                return y[:, :-1, :], c, I_norm, x, out_tot, y_raw
            return y[:, :-1, :], c, I_norm, x, out_tot
        if return_raw_out:
            return y[:, :-1, :], c, I_norm, y_raw 
        return y[:, :-1, :], c, I_norm
    
    def to(self, device):
        self.device = device
        self.rnn.to(device)
        self.C.to(device)
        return self


class DynAgentLoss(nn.Module):
    def __init__(self, config): 
        super(DynAgentLoss, self).__init__()
        self.config = config
        self.loss_compound_package = config.get("loss_compound_func_package", "utils.loss_helpers")
        self.loss_compound_str = config.get("loss_compound_func", "interpolate_exp")
        module = importlib.import_module(self.loss_compound_package)
        self.loss_compound_fn = getattr(module, self.loss_compound_str)

        self.weight_regularisation_loss = None
        self.weight_regularisation_weight = config.get("weight_regularisation_weight", 0.01)
        self.output_norm_loss = None
        self.output_norm_weight = config.get("output_norm_weight", 0.01)
        self.input_norm_loss = None
        self.compounded_choice_loss = None
        self.input_norm_weight = config.get("input_norm_weight", 0.01)
        self.N = config.get("N", 10)
        self.loss_compound_vector = self.loss_compound_fn(self.N)
    
    def forward(self, y, y_hat, I_norm, params_norm):
        self.weight_regularisation_loss = self.weight_regularisation_weight * (params_norm)
        self.output_norm_loss = self.output_norm_weight * torch.norm(y)
        self.input_norm_loss = self.input_norm_weight * (I_norm)
        # print((nn.functional.binary_cross_entropy(y_hat, y, reduction = "none").reshape(-1, self.N) @ self.loss_compound_vector.to(y.device)).shape)
        self.compounded_choice_loss = sum(nn.functional.binary_cross_entropy(y_hat, y, reduction = "none").reshape(-1, self.N) @ self.loss_compound_vector.to(y.device))/y.shape[1]
        # print(self.compounded_choice_loss)
        # print(f"Weight regularisation: {self.weight_regularisation_loss.shape}")
        # print(f"Output norm: {self.output_norm_loss.shape}")
        # print(f"Input norm: {self.input_norm_loss.shape}")
        # print(f"Compounded choice: {self.compounded_choice_loss.shape}")

        return self.compounded_choice_loss + self.weight_regularisation_loss + self.output_norm_loss + self.input_norm_loss
    
    def write_loss(self, summary_writer, index, stage): 
        summary_writer.add_scalar(f"{stage} WeightRegularisation", self.weight_regularisation_loss, index)
        summary_writer.add_scalar(f"{stage} OutputNorm", self.output_norm_loss, index)
        summary_writer.add_scalar(f"{stage} InputNorm", self.input_norm_loss, index)
        summary_writer.add_scalar(f"{stage} CompoundedChoice", self.compounded_choice_loss, index)
        summary_writer.add_scalar(f"{stage} Loss", self.weight_regularisation_loss + self.output_norm_loss + self.input_norm_loss + self.compounded_choice_loss, index)

class Two_step_torch:
    '''Basic two-step task without choice at second step.'''
    def __init__(self, com_prob = 0.8, rew_gen = 'blocks', 
                 block_length = 50, probs = [0.2, 0.8], step_SD = 0.1):
        assert rew_gen in ['walks', 'fixed', 'blocks', 'trans_rev'], \
            'Reward generator type not recognised.'
        self.com_prob = com_prob  # Probability of common transition.    
        self.rew_gen = rew_gen    # Which reward generator to use.
        if rew_gen in ('blocks', 'trans_rev'):

            self.block_length = block_length # Length of each block.               
            self.probs = probs               # Reward probabilities within block.
        elif rew_gen == 'walks':
            self.step_SD = step_SD # Standard deviation of random walk step sizes.
        elif rew_gen == 'fixed':
            self.probs = probs

    def reset(self, n_blocks=100, n_trials = 1000):
        'Generate a fresh set of reward probabilities.'
        if self.rew_gen == 'walks':
            self.reward_probs  = _gauss_rand_walks_torch(n_blocks, n_trials, self.step_SD)
        elif self.rew_gen == 'fixed':
            self.reward_probs  = _const_reward_probs_torch(n_blocks, n_trials, self.probs)
        elif self.rew_gen == 'blocks':
            self.reward_probs  = _fixed_length_blocks_torch(n_blocks, n_trials, self.probs, self.block_length)
        elif self.rew_gen == 'trans_rev': # Version with reversals in transition matrix.
            #self.reward_probs  = _fixed_length_blocks(n_trials, self.probs, self.block_length * 2)
            #self.trans_probs = _fixed_length_blocks(n_trials + self.block_length,
            #                   self.probs, self.block_length * 2)[self.block_length:,0]
            self.reward_probs = _fixed_length_blocks_torch(n_blocks, n_trials, self.probs, self.block_length)
            self.trans_probs = _fixed_length_blocks_torch(n_blocks, n_trials, self.probs, self.block_length * 5)[:,0]
            self.trans_prob_iter = iter(self.trans_probs.permute(1, 0, 2))
        self.rew_prob_iter = iter(self.reward_probs.permute(1, 0, 2))


    def trial(self, choice):
        'Given first step choice generate second step and outcome.'
        if self.rew_gen == 'trans_rev':
            self.com_prob = next(self.trans_prob_iter)
        if isinstance(self.com_prob, float):
            self.com_prob = torch.tensor(self.com_prob, device=choice.device).repeat(choice.shape[0])
        if self.com_prob.shape[0] != choice.shape[0]: 
            self.com_prob = torch.tensor(self.com_prob[0], device=choice.device).repeat(choice.shape[0])

        # print(self.com_prob.shape)
        transition  = torch.tensor((torch.rand_like(self.com_prob) <  self.com_prob), dtype = torch.int, device = choice.device)   # 1 if common, 0 if rare.
        second_step = torch.tensor((choice == transition), dtype = torch.int, device = choice.device)        # Choice 1 (0) commonly leads to second_step 1 (0).
        rew_prob_iter = next(self.rew_prob_iter).to(choice.device)
        # print(self.com_prob.shape)
        # print(rew_prob_iter[:,second_step].shape)
        outcome = torch.tensor((torch.rand_like(self.com_prob) < torch.tensor([rew_prob_iter[i,s] for i, s in enumerate(second_step)], device = choice.device)), dtype = torch.int, device = choice.device)
        correct_choice = torch.tensor([0 if rew_prob_iter[i,0] > rew_prob_iter[i,1] else 1 for i, s in enumerate(second_step)], device = choice.device)
        return (second_step, outcome, correct_choice)


