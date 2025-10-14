"""
RTRNN.py

This module defines recurrent neural network architectures for modeling both reaction time (RT) and discrete choice behavior in cognitive tasks. 

Key classes:
- RT_output_node: Output head for models predicting both RT and choice probabilities.
- RTRNN: A recurrent network for joint RT and choice prediction, supporting flexible output heads and input preprocessing.
- RTSlowFastRNN: A two-timescale RNN with slow and fast cores, designed for tasks with temporally structured evidence accumulation and decision boundaries.
- RTDiscretisedRNN: A discretized RNN for modeling trial structure with explicit reward, fixation, and action periods, and for generating/learning from discretized action trains.

See Jupyter notebooks for examples: e.g., sim2test.ipynb, RNN_RT_discretised.ipynb), 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .BaseNNAgent import BaseNNAgent
from torch.distributions import MultivariateNormal as Normal
import copy


class RT_output_node(nn.Module):
    """
    Created for outputting RT distributions. Here it's a mixed Gaussian model to be parametrized. 
    Mixing probabilities would specify marginal probabilities of each choice. 
    The Gaussian parameters (only two mu-s here, sigma fixed to 1 at this point) specify the log(RT) distribution per choice.
    """

    def __init__(self, input_size, output_size):
        # input size is whatever the hidden feature looks like from previous layers
        # output size is different from elsewhere specified - TODO make this thing consistent!
        # output size tells you the dimensionality of the discrete choice space.

        super(RT_output_node, self).__init__()
        self.mu = nn.Linear(input_size, output_size)
        self.alpha = nn.Linear(input_size, output_size)

    def forward(self, x):
        mu = torch.exp(self.mu(x))
        alpha = nn.functional.softmax(self.alpha(x), dim=-1)
        return torch.cat((mu, alpha), dim=-1)


class RTRNN(nn.Module):
    """
    A generalized version of the RTify architecture thing. Here choice and RT are optimized functions of the RNN hidden states progression.
    Note that here, the RT output defaults to the log(RT) in seconds.  
    """
    def __init__(self, model_params):
        super(RTRNN, self).__init__()
        self.recurrence_per_trial = model_params.get("recurrence_per_trial", 1)
        self.input_size = model_params.get("input_size", 4)  # inject input size
        self.output_size = model_params.get("output_size", 2)
        self.hidden_size = model_params.get("hidden_size", 64)
        self.hidden_output = model_params.get("trial_output_hidden", 0)
        self.pad_zeros = model_params.get("pad_zeros", 1)
        self.last_step = model_params.get("last_step", False) # whether to just use the last step of the hidden train to make predictions. 
        self.out_distribution = model_params.get("out_distribution", False) # this specifies if we output parameters for a particular distribution of the RT (else just a point estimate on the log(RT)). I thought it was quite good, but not using it seemed to work just as well so NOTE I never tried it since a first-pass design of the codes. 
        self.noisy = model_params.get("noisy", False)
        # NOTE: "noisy" not implemented yet
        self.trial_interpolation = model_params.get("trial_interpolation", False)
        self.input_size += int(self.trial_interpolation)

        def build_sequential():
            seq = []
            if self.last_step:
                seq.append(nn.Linear(self.hidden_size, self.hidden_dimensions[0]))
            else:
                seq.append(
                    nn.Linear(
                        self.recurrence_per_trial * self.hidden_size,
                        self.hidden_dimensions[0],
                    )
                )
            seq.append(getattr(nn, self.hidden_nonlinearities[0])())
            if len(self.hidden_dimensions) > 1:
                for i, (hidden_dimension, hidden_nonlinearity) in enumerate(
                    zip(self.hidden_dimensions[1:], self.hidden_nonlinearities[1:])
                ):
                    seq.append(
                        nn.Linear(self.hidden_dimensions[i - 1], hidden_dimension)
                    )
                    seq.append(getattr(nn, hidden_nonlinearity)())
            if self.out_distribution:
                seq.append(RT_output_node(self.hidden_dimensions[-1], self.output_size))
            else:
                seq.append(nn.Linear(self.hidden_dimensions[-1], self.output_size))
            return seq

        if self.hidden_output:
            self.hidden_dimensions = model_params.get("hidden_dimensions", [64])
            self.hidden_nonlinearities = model_params.get(
                "hidden_nonlinearities", ["ReLU"]
            )

            self.trial_output = nn.Sequential(*build_sequential())
        else:
            self.hidden_dimensions = None

            if self.out_distribution:
                self.trial_output = RT_output_node(
                    self.recurrence_per_trial * self.hidden_size, self.output_size
                )
            else:
                self.trial_output = nn.Linear(
                    self.recurrence_per_trial * self.hidden_size, self.output_size
                )  # stationary linear output layer: readout choice and RT per recurrence_per_trial steps

        self.cell_type = model_params.get("cell_type", "GRU").upper()
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError("Invalid cell type.")

    def forward(self, x):
        batch_size = x.shape[0]
        self.h0 = torch.randn(1, batch_size, self.hidden_size, device=self.device)
        seq_len = x.shape[1]
        x = self.transform_input(x)
        h, _ = self.rnn(x, self.h0)
        h = h.reshape(
            batch_size, seq_len, self.recurrence_per_trial, self.hidden_size
        )
        h = h.reshape(batch_size * seq_len, -1)
        out = self.trial_output(h)
        if not self.out_distribution:
            out[..., 0] = torch.sigmoid(out[..., 0])  # choice
        out = out.reshape(
            batch_size,
            seq_len,
            self.output_size * (1 + int(self.out_distribution)),
        )
        return out

    def transform_input(self, x):
        """NOTE: here I operate on inputs by repeating or padding 0s"""
        if self.pad_zeros:
            x = torch.cat(
                (
                    x.unsqueeze(2),
                    torch.zeros(
                        x.shape[0],
                        x.shape[-2],
                        self.recurrence_per_trial - 1,
                        x.shape[-1],
                        device=self.device,
                    ),
                ),
                dim=2,
            )
        else:
            x = torch.cat(
                (
                    x.unsqueeze(2),
                    x.unsqueeze(2).repeat(1, 1, self.recurrence_per_trial - 1, 1),
                ),
                dim=2,
            )

        # print(x.shape)
        if self.trial_interpolation:
            t = (
                torch.linspace(0, 1, self.recurrence_per_trial)
                .reshape(1, 1, self.recurrence_per_trial, 1)
                .repeat(*x.shape[:2], 1, 1)
                .to(x.device)
            )
            x = torch.cat([x, t], dim=-1)

        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x

    def to(self, device):
        self.device = device
        return super(RTRNN, self).to(device)

    def show(self, x):
        """Show the output of the network."""
        batch_size = x.shape[0]
        self.h0 = torch.randn(1, batch_size, self.hidden_size, device=self.device)
        seq_len = x.shape[1]
        x = self.transform_input(x)
        h, _ = self.rnn(x, self.h0)
        h_ = copy.deepcopy(h)
        h = h.reshape(batch_size * seq_len, -1)

        out = self.trial_output(h)
        if not self.out_distribution:
            out[..., 0] = torch.sigmoid(out[..., 0])  # choice

        out = out.reshape(
            batch_size,
            seq_len,
            self.output_size * (1 + int(self.out_distribution)),
        )
        return h_, out


class RTSlowFastRNN(nn.Module):
    """
    Implements the RTify architecture (Cheng et al. 2024, NeurIPS)
    Computational flow: 
    - updates slow net (currently just set to be a GRU, but change it to something else later for model comparisons if you'd like)
    - per step, use the project_sf to project the slow states to serve as the initial states of the fast net
    - use RTify to train the fast net, such that the evidence accumulator crosses the trained threshold to determine RT 
    """
    def __init__(self, model_params):
        super().__init__()
        # ---------- hyper‐params (abbrev) ----------
        B = model_params.get;  self.input_size  = B("input_size", 4)
        self.nC           = B("n_classes", 2)
        self.h_slow_dim   = B("h_slow_dim", 128)
        self.h_fast_dim   = B("h_fast_dim", 64)
        self.K            = B("fast_steps", 15) # fast steps
        self.use_id_proj  = B("identity_proj", True) # whether to use identity projection when h_slow_dim == h_fast_dim
        self.trial_output_hidden = B("trial_output_hidden", 0) # whether to use a hidden layer before choice readout
        self.trial_output_hidden_size = B("trial_output_hidden_size", [64,])
        self.trial_output_hidden_nonlinearities = B("trial_output_hidden_nonlinearities", ["ReLU"])
        self.external_sum_evidence = B("external_sum_evidence", False) # whether to explicitly sum evidence (True) or let the network learn to do so (False)

        # ---------- slow core ----------
        self.rnn_slow     = nn.GRU(self.input_size, self.h_slow_dim, batch_first=True)
        # ---------- projection ----------
        self.project_sf   = (nn.Identity() if self.h_slow_dim==self.h_fast_dim
                             and self.use_id_proj else
                             nn.Linear(self.h_slow_dim, self.h_fast_dim))
        # ---------- fast core ----------
        self.rnn_fast     = nn.GRU(1, self.h_fast_dim, batch_first=True)
        # ---------- heads ----------
        self.initial_evidence = nn.Linear(self.h_fast_dim, 1, bias = False)  if B("initial_evidence", True) else None

        self.evidence_head = nn.Sequential(
            nn.Linear(self.h_fast_dim, 1, bias=False),
            # nn.ReLU()
        )
        nn.init.uniform_(self.evidence_head[0].weight, a=0.01, b=0.1)

        if not self.trial_output_hidden:
            self.fast_choice_head = nn.Linear(self.h_fast_dim, 1)     # binary prob
            self.slow_choice_head = nn.Linear(self.h_slow_dim, 1)     # binary prob
        else: 
            fast_layers = []
            slow_layers = []

            for i, (hidden_size, nonlinearity) in enumerate(
                zip(self.trial_output_hidden_size, self.trial_output_hidden_nonlinearities)
            ):
                if i == 0:
                    fast_layers.append(nn.Linear(self.h_fast_dim, hidden_size))
                    slow_layers.append(nn.Linear(self.h_slow_dim, hidden_size))
                else:
                    fast_layers.append(getattr(nn, nonlinearity)())
                    slow_layers.append(getattr(nn, nonlinearity)())
                    fast_layers.append(nn.Linear(self.trial_output_hidden_size[i - 1], hidden_size))
                    slow_layers.append(nn.Linear(self.trial_output_hidden_size[i - 1], hidden_size))

            self.fast_choice_head = nn.Sequential(*fast_layers, nn.Linear(self.trial_output_hidden_size[-1], 1))
            self.slow_choice_head = nn.Sequential(*slow_layers, nn.Linear(self.trial_output_hidden_size[-1], 1))

        # learnable threshold θ
        self.theta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # scalar threshold
        # optional Δt scale for true RT
        self.max_RT = 1.5
        self.delta_t = self.max_RT / self.K  # scale to [0, max_RT] interval

    # -------------------------------------------------------------------------
    def forward(self, x, slow_only = False, return_all=False):
        """
        x : (B, T, input_size)      SAR tuples per trial
        returns dict:
            rt           (B, T, 1)
            slow_choice  (B, T, 1)
            fast_choice  (B, T, 1)
            phi          (B, T, K)
            logits_all   (B, T, K, 1)
        """
        B, T, _ = x.shape
        device  = x.device
        zeros_K = torch.zeros(B, self.K, 1, device=device)

        # -------- slow dynamics over trials -------------------------------
        h0_slow = torch.zeros(1, B, self.h_slow_dim, device=device)
        slow_out, _ = self.rnn_slow(x, h0_slow)         # (B,T,h_slow)
        if slow_only:
            # if only slow dynamics are needed, return early
            slow_p = torch.sigmoid(self.slow_choice_head(slow_out))
            return dict(slow_choice=slow_p)

        # containers
        RT, slow_choice, fast_choice = [], [], []
        phi_all, logit_traj = [], []
        if return_all: 
            fast_out_all = []
            

        for t in range(T):
            h_s = slow_out[:, t, :]                                     # (B,h_slow)
            slow_p = torch.sigmoid(self.slow_choice_head(h_s))          # (B,1)

            h0_f = self.project_sf(h_s).unsqueeze(0)                    # (1,B,h_fast)
            fast_out, _ = self.rnn_fast(zeros_K, h0_f)                  # (B,K,h_fast)
            if return_all:
                fast_out_all.append(fast_out)

            phi   = self.evidence_head(fast_out).squeeze(-1)            # (B,K)
            phi_0 = phi[:, 0] + self.initial_evidence(fast_out[:,0,:]).squeeze(-1)  if self.initial_evidence is not None else 0# (B,K)
            phi = torch.cat([phi_0.unsqueeze(1), phi[:, 1:]], dim=1)
            if self.external_sum_evidence:
                phi = torch.cumsum(phi, dim=1)                          # (B,K)

            logits= self.fast_choice_head(fast_out).squeeze(-1)         # (B,K)

            # ------------- first passage time ----------------------------
            tau_hat = FirstPassageSTE.apply(phi, self.theta)            # (B,)
            RT.append(tau_hat.unsqueeze(-1) * self.delta_t)

            # gather fast choice at τ
            idx = tau_hat.long().clamp(max=self.K-1).unsqueeze(-1)      # (B,1)
            p_fast = torch.sigmoid(logits.gather(1, idx))               # (B,1)

            slow_choice.append(slow_p)
            fast_choice.append(p_fast)

            phi_all.append(phi)
            logit_traj.append(logits.unsqueeze(-1))                     # keep for dbg

        # -------- stack over trials --------------------------------------
        rt    = torch.stack(RT,          dim=1)   # (B,T,1)
        sc    = torch.stack(slow_choice, dim=1)   # (B,T,1)
        fc    = torch.stack(fast_choice, dim=1)   # (B,T,1)
        phi   = torch.stack(phi_all,     dim=1)   # (B,T,K)
        logits_all = torch.stack(logit_traj, dim=1)  # (B,T,K,1)
        fast_out_all = torch.cat(fast_out_all, dim=1) if return_all else None  # (B,T,K,h_fast)

        if not return_all:
            return dict(rt=rt,
                        slow_choice=sc,
                        fast_choice=fc,
                        phi=phi,
                        logits=logits_all)
        else: 
            return dict(rt=rt,
                        slow_choice=sc,
                        fast_choice=fc,
                        phi=phi,
                        logits=logits_all,
                        slow_out=slow_out,
                        fast_out=fast_out_all)
        
    def freeze_slow(self): 
        """Freeze the slow core parameters."""
        for param in self.rnn_slow.parameters():
            param.requires_grad = False
        for param in self.project_sf.parameters():
            param.requires_grad = False
        for param in self.slow_choice_head.parameters():
            param.requires_grad = False

    def unfreeze_slow(self): 
        """Unfreeze the slow core parameters."""
        for param in self.rnn_slow.parameters():
            param.requires_grad = True
        for param in self.project_sf.parameters():
            param.requires_grad = True
        for param in self.slow_choice_head.parameters():
            param.requires_grad = True


class FirstPassageSTE(Function):
    """
    The first-passage-time for gradients. 

    Straight-through estimator for
        τ̂ = min{ t : Φ_t > θ }   (integer index, float in graph)

    Forward:
        Φ      (B, K)
        θ      scalar *or* (B, 1)
    Returns:
        τ̂      (B,)  float32
    Backward:
        ∂τ/∂Φ_t = -1/ΔΦ  at first-crossing step
        ∂τ/∂θ   = +1/ΔΦ
        (0-gradient for samples that never cross)
    """
    @staticmethod
    def forward(ctx, Phi, theta):
        B, K = Phi.shape

        # find first crossing index (K   if never crosses)
        above    = (Phi > theta).to(torch.int64)          # (B,K)
        sentinel = torch.ones(B, 1, dtype=above.dtype, device=Phi.device)
        idx      = torch.argmax(torch.cat([above, sentinel], dim=1), dim=1)  # (B,)

        ctx.save_for_backward(Phi, theta, idx)
        ctx.K = K
        return idx.float()                                # keeps graph

    @staticmethod
    def backward(ctx, grad_tau):
        Phi, theta, idx_full = ctx.saved_tensors
        K = ctx.K

        # mask for "no crossing" samples
        no_cross = idx_full == K
        idx      = idx_full.clone().clamp(max=K-1)        # safe for gather
        idx_prev = (idx - 1).clamp(min=0)

        delta_mask_eps = 1e-8 * (idx == 0).to(Phi.dtype)  # set up a mask for situations where idx = 0

        # ΔΦ = Φ_t − Φ_{t-1}
        Delta = Phi.gather(1, idx.unsqueeze(1)).squeeze(1) - \
                Phi.gather(1, idx_prev.unsqueeze(1)).squeeze(1) + delta_mask_eps

        # avoid div-by-0 and suppress grads if no crossing
        Delta   = Delta.masked_fill(no_cross, 1.0)        # dummy denom
        scale   = (-grad_tau / Delta).masked_fill(no_cross, 0.0)  # (B,)

        # gradient wrt Φ : only at idx (0 if no_cross)
        g_Phi = torch.zeros_like(Phi)
        g_Phi.scatter_(1, idx.unsqueeze(1), scale.unsqueeze(1))

        # gradient wrt  θ
        g_theta = (grad_tau / Delta).masked_fill(no_cross, 0.0)
        g_theta = g_theta.sum().unsqueeze(0) if theta.ndim == 0 else g_theta.unsqueeze(1)
        # print("g_theta:", g_theta)

        return g_Phi, g_theta


class RTDiscretisedRNN(nn.Module): 

    """ 
    Replicates in spirit ideas in the Task-DyVA paper 
    - replaces the Bayesian DS with a simple GRU model 
    - on each timestep there are three possible actions: 
        1. no action
        2. choice L
        3. choice R

    - this method also allows us to really dissect the sub-trial dynamics

    trial structure outlines as: 
        - reward period (fixation set to 1; trial feedback presented)
        - fixation period (fixation set to 1)
        - action period (fixation set to 0)

    you get to specify the length of each period. 

    """
    
    def __init__(self, model_params):
        super(RTDiscretisedRNN, self).__init__()
        self.dt = model_params.get("dt", 0.1) # in seconds
        self.reward_presentation_time = model_params.get("reward_presentation_time", 0.5) # in seconds
        self.fixation_time = model_params.get("fixation_time", 1) # in seconds
        self.action_time = model_params.get("action_time", 1.5) # in seconds, also RT upper limit
        self.repeat_stimulus_during_fixation = model_params.get("repeat_stimulus_during_fixation", 1) \
            # this is about whether we repeat the stim presented at fixation over in the fixation period
        self.repeat_stimulus_during_action = model_params.get("repeat_stimulus_during_action", 1)
        self.sigma_action = model_params.get("sigma_action", 0.1)
        self.pad_ones = model_params.get("pad_ones", 0) # whether to pad ones in the action period after an action is observed

        self.one_hot_input = model_params.get("one_hot_input", False)

        self.input_size = 4 if not self.one_hot_input else 7 \
            # encoding options of the input space; note this changes the hypothesis classes. 
        self.output_size = model_params.get("output_size", 3) 

        self.hidden_size = model_params.get("hidden_size", 64)
        self.hidden_output = model_params.get("trial_output_hidden", 0)
        self.train_h0 = model_params.get("train_h0", False)
 
        self.noisy = model_params.get("noisy", False)
        # NOTE: "noisy" not implemented

        def build_sequential():
            seq = []
            
            seq.append(nn.Linear(self.hidden_size, self.hidden_dimensions[0]))

            seq.append(getattr(nn, self.hidden_nonlinearities[0])())
            if len(self.hidden_dimensions) > 1:
                for i, (hidden_dimension, hidden_nonlinearity) in enumerate(
                    zip(self.hidden_dimensions[1:], self.hidden_nonlinearities[1:])
                ):
                    seq.append(
                        nn.Linear(self.hidden_dimensions[i - 1], hidden_dimension)
                    )
                    seq.append(getattr(nn, hidden_nonlinearity)())
                
            seq.append(nn.Linear(self.hidden_dimensions[-1], self.output_size))
            return seq

        if self.hidden_output:
            self.hidden_dimensions = model_params.get("hidden_dimensions", [64])
            self.hidden_nonlinearities = model_params.get(
                "hidden_nonlinearities", ["ReLU"]
            )

            self.trial_output = nn.Sequential(*build_sequential())

        else:
            self.hidden_dimensions = None
            self.trial_output = nn.Linear(
                self.hidden_size, self.output_size
            )  # stationary linear output layer: readout choice and RT per recurrence_per_trial steps

        self.cell_type = model_params.get("cell_type", "GRU").upper()
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError("Invalid cell type.")
        if self.train_h0:
            self.h0 = nn.Parameter(torch.randn(1, self.hidden_size))
        else: 
            self.h0 = None
        

    def forward(self, x, softmax = False, return_h = False):
        if not self.train_h0:
            h0 = torch.randn(1, x.shape[0], self.hidden_size, device=self.device)
        else: 
            h0 = self.h0.unsqueeze(0).repeat(1, x.shape[0], 1).to(self.device)
        
        seq_len = x.shape[1]
      
        h, _ = self.rnn(x,h0)
        out = self.trial_output(h)
        if softmax:
            out = torch.softmax(out, dim = -1)

        if return_h:
            return out, h
        return out

    def to(self, device):
        self.device = device
        return super(RTDiscretisedRNN, self).to(device)

    def convert_dataset(self, u, x): 
        """
        Converts a dataset from standard form to the format in this context!
        trial structure outlines as: 
        - reward period (fixation set to 1; trial feedback presented)
        - fixation period (fixation set to 1)
        - action period (fixation set to 0)

        algorithm: 
        - produce an action-time train. 
            create a (B, T_trials, self.action_time // self.dt, 3) tensor
            populate the 1st and 2nd elements of the last dimension (L and R actions) at the rt // dt time.
            smooth each mini-traj within a trial with self.action_sigma
            populate the 0th dimension with 1 - sum(). 
            - the stim for the train shall be (B, T_trials, self.action_time // self.dt, self.input_size)
            - if self.repeat_stimulus_during_action, the first (self.input_size-1) elements in the 2nd dimension 
                should repeat what's in u. Fixation (the last element) set to 0. if self.one_hot_input, this shall
                be 6 + 1 elements (each option be an element). Else, this shall be 3 + 1 elements, taking the 0th, 
                2nd and 4th of the u. 
        - similarly produce the fixation_time train. Here, the input should follow the same rule as the action_time 
            train, but fixation set to  1. 
        - similarly produce the reward_time train. Here input presentation is by default. 

        - then just cat them as the trial structure, and reshape for a long time series. 

        
        Args: 
            u: Tensor (B, T_trials, D_in) -- one-hot
            x: Tensor (B, T_trials, 2) -- (choice, rt)
        
        Returns: 
            u: Tensor (B, discretised_time, self.input_size) 
            x: Tensor (B, discretised_time, self.output_size = 3) -- (no_action, L, R). This is Gaussianed. 
        """
        device = getattr(self, "device", u.device)
        u = u.to(device)
        x = x.to(device)

        B, T_trials, D_in = u.shape
        dt = float(self.dt)
        n_reward  = int(round(self.reward_presentation_time / dt))
        n_fix     = int(round(self.fixation_time / dt))
        n_action  = int(round(self.action_time / dt))
        seg_total = n_reward + n_fix + n_action

        # Sanity: infer stimulus-only width implied by input_size
        if self.one_hot_input:
            stim_width = 6
            if self.input_size != 7:
                raise ValueError("With one_hot_input=True, expected input_size==7 (6 stim + fixation).")
        else:
            stim_width = 3
            if self.input_size != 4:
                raise ValueError("With one_hot_input=False, expected input_size==4 (3 stim + fixation).")

        # Prepare output tensors
        U = torch.zeros(B, T_trials, seg_total, self.input_size, device=device, dtype=u.dtype)
        Y = torch.zeros(B, T_trials, seg_total, 3,              device=device, dtype=u.dtype)

        # Build per-segment input templates
        # Map stim per-trial to (B, T, S) without fixation
        stim3or6 = _map_stim_channels(u, one_hot_input=self.one_hot_input)  # (B, T, S)

        # --- Reward period inputs (fixation=1; no stimulus presented by default) ---
        # If you prefer to carry over the previous trial's feedback signal, you could add a channel;
        # here we follow the spec and keep stimulus at zeros.
        if n_reward > 0:
            U[:, :, :n_reward, -1] = 1.0  # fixation channel = 1
            U[:, :, :n_reward, :stim_width] = (
                stim3or6.unsqueeze(2).expand(B, T_trials, n_reward, stim_width)
            )

        # --- Fixation period inputs ---
        if n_fix > 0:
            start = n_reward
            end   = n_reward + n_fix
            U[:, :, start:end, -1] = 1.0  # fixation = 1
            if self.repeat_stimulus_during_fixation:
                # broadcast stim across fixation timesteps
                # U[..., :stim_width] shape: (B,T,n_fix,S)
                U[:, :, start:end, :stim_width] = (
                    stim3or6.unsqueeze(2).expand(B, T_trials, n_fix, stim_width)
                )

        # --- Action period inputs ---
        if n_action > 0:
            start = n_reward + n_fix
            end   = seg_total
            U[:, :, start:end, -1] = 0.0  # fixation = 0 during action
            if self.repeat_stimulus_during_action:
                U[:, :, start:end, :stim_width] = (
                    stim3or6.unsqueeze(2).expand(B, T_trials, n_action, stim_width)
                )
            else:
                # stimuli zeroed by default; nothing to do
                pass

        # --- Targets Y: (no_action, L, R) ---
        # Default everywhere: no_action=1
        Y[..., 0] = 1.0

        # In the action window, place an impulse at floor(rt/dt) in L/R and smooth.
        if n_action > 0:
            # Extract (choice, rt)
            choice = x[..., 0].long().clamp(min=0, max=1)  # 0=L, 1=R
            rt     = x[..., 1]

            # Replace NaN or invalid RT with ceiling
            bad = torch.isnan(rt) | (rt <= 0)
            if bad.any():
                # Follow your note: replace with RT ceiling, and (optionally) print once.
                print(f"[convert_dataset] Found {bad.sum().item()} NaN/invalid RT; replacing with action_time ceiling.")
                rt = torch.where(bad, torch.tensor(self.action_time, device=device, dtype=rt.dtype), rt)

            # Clip to [0, action_time]
            rt = torch.clamp(rt, 0.0, self.action_time)

            # Compute time indices inside the action segment
            t_idx = torch.round(rt / dt).long()
            t_idx = torch.clamp(t_idx, 0, n_action - 1)  # ensure valid index

            # Write impulses in L/R channels at (action start + t_idx)
            act_start = n_reward + n_fix
            if self.pad_ones:
                print('pad ones detected, converting sigma_action to 0')
                self.sigma_action = 0.0
            for b in range(B):
                for t in range(T_trials):
                    idx = act_start + t_idx[b, t].item()
                    ch  = 1 + choice[b, t].item()  # 1=L, 2=R
                    if self.pad_ones:
                        Y[b, t, idx:, ch] = torch.ones_like(Y[b, t, idx:, ch])

                    else: 
                        Y[b, t, idx, ch] = 1.0

            # Smooth L/R with Gaussian along time within the action window only
            sigma_steps = max(self.sigma_action / max(dt, 1e-6), 1e-6)
            kernel = _gaussian_kernel_1d(sigma_steps, device=device, dtype=U.dtype)  # (1,1,K)

            # Slice the action portion: (B,T,n_action,3) -> take L/R -> (B*T, 1, n_action) per channel
            LR = Y[:, :, act_start:act_start + n_action, 1:3]  # (B,T,n_action,2)

            # Convolve each channel separately (groups=2)
            LR_bt = LR.reshape(B * T_trials, n_action, 2).permute(0, 2, 1)  # (B*T,2,n_action)
            # pad='same' via padding=(K-1)//2
            pad = (kernel.shape[-1] - 1) // 2
            smoothed = F.conv1d(LR_bt, kernel.expand(2, 1, -1), padding=pad, groups=2)  # (B*T,2,n_action)
            LR_sm = smoothed.permute(0, 2, 1).reshape(B, T_trials, n_action, 2)


            # Write back smoothed L/R into Y
            Y[:, :, act_start:act_start + n_action, 1:3] = LR_sm

            # Recompute no_action = 1 - (L+R), clipped to [0,1], but ONLY inside the action window
            sum_LR = LR_sm.sum(dim=-1, keepdim=True)  # (B,T,n_action,1)
            no_action = torch.clamp(1.0 - sum_LR, min=0.0, max=1.0)
            Y[:, :, act_start:act_start + n_action, 0:1] = no_action

        # Finally, collapse trials into a long sequence
        U_long = U.reshape(B, T_trials * seg_total, self.input_size)
        Y_long = Y.reshape(B, T_trials * seg_total, 3)
        return U_long, Y_long        


def _gaussian_kernel_1d(sigma_steps: float, device, dtype):
    # make kernel length ~ 6*sigma + 1 (odd)
    half = int(max(1, round(3.0 * sigma_steps)))
    xs = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (xs / max(sigma_steps, 1e-6))**2)
    kernel = kernel / kernel.max()
    return kernel.view(1, 1, -1)  # (out_ch=1, in_ch=1, K)

def _map_stim_channels(u_trial, one_hot_input: bool):
    """
    u_trial: (..., D_in) one-hot stimulus per trial (no fixation channel here)
    Returns: (..., S) where S is 3 or 6 (stim channels only, no fixation)
    Spec:
      - if one_hot_input: take 6 stimulus options (first 6 dims)
      - else: take indices 0, 2, 4  (3 stimulus channels)
    """
    if one_hot_input:
        if u_trial.size(-1) < 6:
            raise ValueError("Expected >=6 stimulus dims in u when one_hot_input=True.")
        return u_trial[..., :6]
    else:
        take = [0, 2, 4]
        if u_trial.size(-1) <= max(take):
            raise ValueError("u has fewer than 5 dims; cannot take indices [0,2,4].")
        return torch.stack([u_trial[..., i] for i in take], dim=-1)
