import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


class HMMModel(nn.Module):
    """
    Base class for the hidden markov model for the cognitive task
    """
    def __init__(self, 
                z_dim: int,  # number of hidden states
                stim_types: int, # number of stimulus types
                symmetric: bool = False,  # whether to enforce symmetry in the model
                 ):
        super(HMMModel, self).__init__()
        self.K = None
        self.P = None
        self.pi = None
        self.z_dim = z_dim
        self.stim_types = stim_types
        self.S = 2 ** (stim_types)
        self.altered = False
        self.fixP = False  # whether to fix the emission probabilities
        self.fixpi = False  # whether to fix the initial state distribution
        self.fixK = False  # whether to fix the transition probabilities
        self.symmetric = symmetric
        self.mirror_idx = build_mirror_idx(self.z_dim) if symmetric else None

    def forward_backward(self, s, x, mask): 
        """
        Forward-backward algorithm for the HMM

        Args:
            s : (batch_size, n_steps) -- the stimulus number at each time index. 
            x : (batch_size, n_steps) -- observed behavior.  
            mask : (batch_size, n_steps) -- mask for the valid time steps

        Returns: 
            gamma : (batch_size, n_steps, z_dim) -- the posterior probability of the hidden state at each time index
            xi : (batch_size, n_steps, z_dim, z_dim) -- the joint probability of hidden progressions
        """
        if x.dim() == 3: 
            x = x.squeeze(-1)
        if s.dim() == 3:
            s = s.squeeze(-1)

        B, T = x.shape

        logpi = torch.log(self.pi) # z_dim
        logP = torch.log(self.P) # z_dim, 3
        logK_unbatched = torch.log(self.K) # the shape would be (S + 1, z_dim, z_dim)
        logK = logK_unbatched[s.squeeze(-1)] # (B, T, z_dim, z_dim)
        
        log_alpha = torch.zeros((B, T, self.z_dim), device = x.device) - 1e10
        # log_alpha[:, 0, :] = logpi.unsqueeze(0) + logP[:, x[:, 0].long()].permute(1, 0) # (B, z_dim)
        log_alpha_prev = logpi.unsqueeze(0)# + logP[:, x[:, 0].long()].permute(1, 0) # (B, z_dim)

        for t in range(0, T):
            if t > 0: 
                log_alpha_prev = log_alpha[:, t-1, :]  # (B, z_dim)
            log_alpha[:, t, :] = ((log_alpha_prev.unsqueeze(1) + logK[:, t, :, :]).logsumexp(dim=-1) \
                + logP[:, x[:, t].long()].permute(1, 0)) * mask[:, t].unsqueeze(-1) \
                    + log_alpha_prev * (1 - mask[:, t]).unsqueeze(-1)
                    # this is of shape (B, z_dim)
        
        loglik = torch.logsumexp(
            log_alpha[:, -1, :], 
            dim=-1
        )

        # Backward pass
        log_beta = torch.zeros((B, T, self.z_dim), device = x.device) - 1e10
        log_beta[:, -1, :] = 0

        valid_next = -1
        logP_next = logP[:, x[:, -1].long()].permute(1, 0)  # (B, z_dim, 1)
        logK_next = logK[:, -1, :, :]
        for t in range(T-2, -1, -1):
            log_beta_next = log_beta[:, t + 1, :] # (B, z_dim)

            
            # log_beta[:, t, :] = (logP[:, x[:, t + 1].long()].permute(1, 0).unsqueeze(-1)\
            #     +logK[:, t +1, :, :] \
            #     + log_beta_next.unsqueeze(-1)).logsumexp(dim=1) \
            #     * mask[:, t ].unsqueeze(-1) \
            #     + log_beta_next * (1 - mask[:, t]).unsqueeze(-1)
            log_beta[:, t, :] = (logP_next.unsqueeze(-1) + logK_next + log_beta_next.unsqueeze(-1)).logsumexp(dim=1) \
                * mask[:, t + 1].unsqueeze(-1) \
                + log_beta_next * (1 - mask[:, t + 1]).unsqueeze(-1)

            logP_next = logP[:, x[:, t].long()].permute(1, 0) * mask[:, t].unsqueeze(-1) + \
                logP_next * (1 - mask[:, t]).unsqueeze(-1)
            logK_next = logK[:, t, :, :] * mask[:, t].unsqueeze(-1).unsqueeze(-1) + \
                logK_next * (1 - mask[:, t]).unsqueeze(-1).unsqueeze(-1)

        ninf = -1e10
        
        log_beta -= loglik.unsqueeze(-1).unsqueeze(-1)  # normalize the backward probabilities

        # Compute posterior probabilities
        log_gamma = log_alpha + log_beta # (B, T, z_dim)
        log_gamma -= log_gamma.logsumexp(dim=-1, keepdim=True)

        # Compute joint probabilities
        log_xi = logK[:, :-1] + log_alpha[:, :-1, :, None] \
                + log_beta[:, 1:, None, :] 
        log_xi = log_xi \
                + logP[:, x[:, 1:].long()].permute(1, 2, 0)[:, :, None, :] # (B, T-1, z_dim)

        log_xi = log_xi - torch.logsumexp(log_xi.view(B, T-1, -1), dim=-1)[:, :, None, None]
        
        # mask out padding
        log_gamma = log_gamma.masked_fill((mask == 0).unsqueeze(-1), ninf)  # mask out padding
        log_xi = log_xi.masked_fill((mask[:, 1:] == 0).unsqueeze(-1).unsqueeze(-1), ninf)


        # log_gamma = log_gamma * mask.unsqueeze(-1)
        # log_xi    = log_xi * mask[:, 1:].unsqueeze(-1).unsqueeze(-1)
        return log_gamma, log_xi, log_alpha, log_beta, loglik

    def predictive_forward(self, s, x, mask):
        """
        Perform the predictive forward pass for the HMM model

        Args:
            s : (batch_size, n_steps) -- the stimulus number at each time index. 
            x : (batch_size, n_steps) -- observed behavior.  
            mask : (batch_size, n_steps) -- mask for the valid time steps

        Returns: 
            log_alpha_: (batch_size, n_steps, z_dim) -- the forward probabilities
        """
        # if self.altered:
        self.get_params()
        if x.dim() == 3: 
            x = x.squeeze(-1)
        if s.dim() == 3:
            s = s.squeeze(-1)

        B, T = x.shape

        logpi = torch.log(self.pi) # z_dim
        logP = torch.log(self.P) # z_dim, 3
        logK_unbatched = torch.log(self.K) # the shape would be (S + 1, z_dim, z_dim)
        logK = logK_unbatched[s.squeeze(-1)] # (B, T, z_dim, z_dim)
        
        log_alpha = torch.zeros((B, T, self.z_dim), device = x.device) - 1e10
        # log_alpha[:, 0, :] = logpi.unsqueeze(0) + logP[:, x[:, 0].long()].permute(1, 0) # (B, z_dim)
        log_alpha_prev = logpi.unsqueeze(0)# + logP[:, x[:, 0].long()].permute(1, 0) # (B, z_dim)

        for t in range(0, T):
            log_alpha[:, t, :] = ((log_alpha_prev.unsqueeze(1) + logK[:, t, :, :]).logsumexp(dim=-1) \
                + logP[:, x[:, t].long()].permute(1, 0)) * mask[:, t].unsqueeze(-1) \
                    + log_alpha_prev * (1 - mask[:, t]).unsqueeze(-1)
                    # this is of shape (B, z_dim)       
            log_alpha_prev = log_alpha[:, t, :]  # update the previous alpha for the next step
            log_alpha[:, t, :] -= logP[:, x[:, t].long()].permute(1, 0) * mask[:, t].unsqueeze(-1)  # remove the emission probabilities from the alpha
        return log_alpha


    def fit_batch(self, s, x, mask):
        """
        Fit the HMM model to the data using the forward-backward algorithm

        Args:
            s : (batch_size, n_steps) -- the stimulus number at each time index. 
            x : (batch_size, n_steps) -- observed behavior.  
            mask : (batch_size, n_steps) -- mask for the valid time steps

        Returns: 
            log_gamma : (batch_size, n_steps, z_dim) -- the posterior probability of the hidden state at each time index
            log_xi : (batch_size, n_steps, z_dim, z_dim) -- the joint probability of hidden progressions
        """
        if self.altered:
            self.get_params()  # ensure the parameters are constructed

        # E step
        with torch.no_grad():
            # run the forward-backward algorithm
            log_gamma, log_xi, _, _, ll = self.forward_backward(s, x, mask)

        # M step
        self.update_params(log_gamma, log_xi, s, x, mask)
        return ll

    def to(self, *args, **kwargs):
        """
        Override the to method to ensure that the parameters are constructed
        """
        super(HMMModel, self).to(*args, **kwargs)

        if self.altered:
            self.get_params()
        self.K = self.K.to(*args, **kwargs)
        self.P = self.P.to(*args, **kwargs)
        self.pi = self.pi.to(*args, **kwargs)
        self.mirror_idx = self.mirror_idx.to(*args, **kwargs) if self.symmetric else None
        return self

    def fix_P(self, P = None): 
        if P is None: 
            self.fixP = True
        else:
            self.P = P
            self.fixP = True
    
    def fix_pi(self, pi = None):
        if pi is None:
            self.fixpi = True
        else:
            self.pi = pi
            self.fixpi = True
    
    def fix_K(self, K = None):
        if K is None:
            self.fixK = True
        else:
            self.K = K
            self.fixK = True

    def save_checkpoint(self, path):
        """
        Save the model parameters to a file.
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "K": self.K,
            "P": self.P,
            "pi": self.pi,
            "fixP": self.fixP,
            "fixpi": self.fixpi,
            "fixK": self.fixK,
            "z_dim": self.z_dim,
            "stim_types": self.stim_types,
        }, path)
    
    def load_checkpoint(self, path):
        """
        Load the model parameters from a file.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.K = checkpoint["K"]
        self.P = checkpoint["P"]
        self.pi = checkpoint["pi"]
        self.fixP = checkpoint["fixP"]
        self.fixpi = checkpoint["fixpi"]
        self.fixK = checkpoint["fixK"]
        self.z_dim = checkpoint["z_dim"]
        self.stim_types = checkpoint["stim_types"]
        
    def enforce_symmetry(self):
        """
        Build *symmetric* (logpi, logP, logK) tensors WITHOUT in-place writes.
        Returns:
            logpi_sym : (K,)
            logP_sym  : (K, X)
            logK_sym  : (S+1, K_to, K_from)
        Assumes:
            self.logK shape == (S+1, K, K) with softmax over dim=-2 (rows: 'to')
            self.mirror_idx  : LongTensor [K] mapping i -> M(i)
            self.action_bit  : int   (use your attr instead of hardcoding 2)
            self.S           : number of real stimuli (last slice is grounding)
        """
        device = self.logpi.device
        m = self.mirror_idx.to(device)

        # ----- π (respect fix flag)
        if getattr(self, "fixpi", False):
            logpi_sym = self.logpi
        else:
            logpi_sym = 0.5 * (self.logpi + self.logpi[m])

        # ----- P (respect fix flag)
        # If your symbols are [L,R,ND], swap L<->R across mirrors:
        emission_perm = getattr(self, "emission_perm", torch.tensor([0, 2, 1], device=device))
        if getattr(self, "fixP", False) or emission_perm is None:
            logP_sym = self.logP
        else:
            perm = emission_perm.to(device)
            logP = self.logP
            cols = []
            C = logP.size(1)
            for c in range(C):
                tgt = int(perm[c])
                avg = 0.5 * (logP[:, c] + logP[m, tgt])
                cols.append(avg.unsqueeze(1))
            logP_sym = torch.cat(cols, dim=1)

        # ----- K: tie stimulus s with its action-flipped partner s^mask
        flip_mask = 1 << getattr(self, "action_bit", 2)  # use your attribute
        logK_sym = self.logK.clone()
        seen = set()
        for s in range(self.S):
            sf = s ^ flip_mask
            key = (min(s, sf), max(s, sf))
            if key in seen:
                continue
            seen.add(key)

            A  = logK_sym[s]                               # (to, from)
            Bm = logK_sym[sf].index_select(0, m).index_select(1, m)
            Aavg = 0.5 * (A + Bm)
            logK_sym[s]  = Aavg
            logK_sym[sf] = Aavg.index_select(0, m).index_select(1, m)

        # grounding slice symmetric by itself
        G  = logK_sym[-1]
        Gm = G.index_select(0, m).index_select(1, m)
        logK_sym[-1] = 0.5 * (G + Gm)

        return logpi_sym, logP_sym, logK_sym

    def get_params(self): 
        """
        NO ARGS: 
            constructs the transition matrix K(s) as an array of matrices
        """
        self.logK = self.logK_base.new_zeros(self.S + 1, self.z_dim, self.z_dim)
        self.logK = self.logK.to(self.logK_base.device)
        self.construct_logK()
        self.logK[-1, :, :] += self.logK_grounding

        if self.symmetric: 
            logpi_sym, logP_sym, logK_sym = self.enforce_symmetry()
        else: 
            logpi_sym = self.logpi
            logP_sym = self.logP
            logK_sym = self.logK

        if not self.fixK:
            self.K = F.softmax(logK_sym, dim=-2)
        if not self.fixP:
            self.P = F.softmax(logP_sym, dim=-1) 
        else: 
            self.logP.data = self.P.log()  # ensure that the logP is in log space
        if not self.fixpi:
            self.pi = F.softmax(logpi_sym, dim=-1)
        else:
            self.logpi.data = self.pi.log()

        self.altered = False
        return self.K, self.P

    def fit_bptt(self, s, x, mask, optim, loss="BCE", val = False): 
        
        if val:
            with torch.no_grad():
                log_alpha_prime = self.predictive_forward(s, x, mask)  # to get a predictive distribution
        else:
            log_alpha_prime = self.predictive_forward(s, x, mask) # to get a predictive distribution
        log_alpha_prime = log_alpha_prime - log_alpha_prime.min(dim=-1, keepdim=True).values  # normalize the forward probabilities
        alpha = torch.exp(log_alpha_prime)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)  # normalize the forward probs
        choice_probs = alpha @ self.P  # (B, T, z_dim) @ (z_dim, 3) = (B, T, 3)

        # optim = torch.optim.Adam(self.parameters(), lr=lr) if optimizer == "adam" else LBFGS(self.parameters(), lr=lr, max_iter=10, line_search_fn='strong_wolfe')
        def loss_fn(loss):
            """
            Compute the loss for the BPTT training
            """
            target = torch.zeros_like(choice_probs, device=choice_probs.device)
            target[:, :, 0] = (x == 0).float()  # no response
            target[:, :, 1] = (x == 1).float()  # response 1
            target[:, :, 2] = (x == 2).float()  # response 2

            if loss == "BCE":
                # binary cross-entropy loss
                l = F.binary_cross_entropy(choice_probs, target.float(), reduction='mean')
            elif loss == "MSE":
                # mean squared error loss
                l = F.mse_loss(choice_probs, target.float(), reduction='mean')
            elif loss == "NLL": 
                # negative log-likelihood loss
                l = -torch.sum(target * torch.log(choice_probs + 1e-12) + (1 - target) * torch.log(1 - choice_probs + 1e-12)) / target.shape[0]
            else:
                raise ValueError(f"Unknown loss function: {loss}")
            return l

        optim.zero_grad()
        l = loss_fn(loss)
        
        if not val:
            l.backward()
            optim.step()
            self.altered = True  # mark the model as altered, so that we can re-compute the parameters next time
            self.get_params()
        return l

class HMMModel_linearK(HMMModel):
    """
    Implements the hidden markov model for the cognitive task, as an alternative to the RNN
    - the LinearK version: constructs the transition matrix linearly per stimulus  
    """
    def __init__(self, 
                z_dim: int, 
                stim_types: int, # number of stimulus types
                symmetric: bool = False,  # whether to enforce symmetry in the model
                ):
        super(HMMModel_linearK, self).__init__(
            z_dim=z_dim, 
            stim_types=stim_types,
            symmetric=symmetric
        )

        self.logK_base = nn.Parameter(torch.randn(z_dim, z_dim))
        self.logK_stim = nn.Parameter(torch.randn(stim_types, z_dim, z_dim))
        self.logK_grounding = nn.Parameter(torch.randn(z_dim, z_dim))
        self.logK = None
        self.logP = nn.Parameter(torch.randn(z_dim, 3))  # emissions. The third bracket gives no-response
        self.logpi = nn.Parameter(torch.randn(z_dim))  # initial state distribution

    def construct_logK(self):
        idx = 0
        for stim in product([0, 1], repeat = self.stim_types):
            stim = torch.tensor(stim).float().to(self.logK_stim.device)
            self.logK[idx, :, :] += (stim @ self.logK_stim.reshape(self.stim_types, -1)).reshape(self.z_dim, self.z_dim)
            idx += 1
   
    def update_params(self, log_gamma, log_xi, s, x, mask):
        """
        M-step for Linear-K.  Updates in-place the nn.Parameters:
        • self.logpi
        • self.logP
        • self.logK_base, self.logK_stim, self.logK_grounding
        """
        B, T, K = log_gamma.shape
        S = self.S                     # 2^{stim_types}
        n_sym = self.logP.size(1)      # number of emission symbols (3)
        alpha_tr = 1 #0.25
        alpha_em = 1e-8 #0.25
        sticky = 2.0

        # -------- closed-form for π and P ---------------------------------

        with torch.no_grad():
            # initial
            N_pi = log_gamma[:, 0, :].exp().sum(dim=0) + 1e-8
            if not self.fixpi:
                self.logpi.copy_(torch.log(N_pi) - torch.log(N_pi.sum()))
            else: 
                self.logpi.copy_(self.pi.log())

            # emission
            N_emit = _soft_counts_emit(log_gamma, x, mask, n_sym) + alpha_em 
            if not self.fixP:
                self.logP.copy_(torch.log(N_emit) - torch.log(N_emit.sum(dim=1, keepdim=True)))
            else:
                self.logP.copy_(self.P.log())

        # -------- expected transition counts per stimulus -----------------

        N_xi = _soft_counts_xi(log_xi, s, mask, S) + alpha_tr         # (S,K,K)
        # print(N_xi[1])
        # raise Exception("check")

        N_xi[:, torch.arange(K), torch.arange(K)] += sticky  # add stickiness to the diagonal
        
        stim_bits = torch.tensor(list(product([0,1], repeat=self.stim_types)),
                                dtype=self.logK_stim.dtype, device=self.logK_stim.device)   # (S,stim_types)

        optim = LBFGS([self.logK_base, self.logK_stim, self.logK_grounding],
                    lr=1.0, max_iter=10, line_search_fn='strong_wolfe')


        # -------- gradient ascent on transition logits -------------------
        def neg_Q():
            """
            Negative expected complete-data log-likelihood
            with respect to transition parameters only.
            """
            zeros1 = torch.zeros_like(self.logK_base).unsqueeze(0)  # (1,K,K)
            zeros2 = torch.zeros(stim_bits.shape[0], K, K, device=self.logK_stim.device)
            K_bank = F.softmax(
                self.logK_base.unsqueeze(0) +                 # (1,K,K)
                torch.cat([torch.einsum('sd,dkj->skj', stim_bits, self.logK_stim), zeros1], dim = 0) +
                torch.cat([zeros2,
                        self.logK_grounding.unsqueeze(0)], dim=0),   # grounding added to last index
                dim=-2)                                        # (S+1,K,K)

            # slice grounding row
            # K_bank[-1] = F.softmax(self.logK_grounding, dim=-2)

            logK_bank = torch.log(K_bank + 1e-12)              # (S+1,K,K)
            ll =  -(N_xi * logK_bank[:S]).sum()                  # expected log-likelihood
            optim.zero_grad()
            ll.backward()
            return ll

        optim.step(neg_Q)

        self.altered = True  # mark the model as altered, so that we can re-compute the parameters next time

        # ---------- refresh K, P, π tensors for next E-step ---------------
        self.get_params() 


                
class HMMModel_symmetric(HMMModel):
    """
    Symmetric HMM with action handled via mirroring, and transitions
    parameterised by reward/state context according to `model_type`:
      - "additive":     B + r*R + s*S
      - "interaction":  B + r*R + s*S + (r&s)*RS
      - "perstate":     separate matrix for each (r,s) in {(0,0),(1,0),(0,1),(1,1)}
    The 'grounding' matrix is separate and symmetric by itself.
    """
    def __init__(self,
                z_dim: int,
                stim_types: int,
                model_type: str = "interaction",
                action_bit: int = 2,
                reward_bit: int = 1,
                state_bit: int  = 0):
        super().__init__(z_dim=z_dim, stim_types=stim_types, symmetric=True)
        assert model_type in {"additive","interaction","perstate"}
        self.model_type = model_type
        self.action_bit = action_bit
        self.reward_bit = reward_bit
        self.state_bit  = state_bit

        K = z_dim
        # --- transitions (logits over 'to', columns 'from') ---
        if model_type in {"additive", "interaction"}:
            self.logK_base = nn.Parameter(torch.randn(K, K))
            self.logK_R    = nn.Parameter(torch.randn(K, K))
            self.logK_S    = nn.Parameter(torch.randn(K, K))
            if model_type == "interaction":
                self.logK_RS = nn.Parameter(torch.randn(K, K))
        else:  # perstate: 4 contexts excluding action
            self.logK_ctx  = nn.Parameter(torch.randn(4, K, K))

        # grounding (trial-end), will be symmetrised by construction
        self.logK_grounding = nn.Parameter(torch.randn(K, K))

        # --- emissions/initial: parameterize only first half, mirror the rest ---
        self.logP_half  = nn.Parameter(torch.randn(K//2, 3))   # [L,R,ND]
        self.logpi_half = nn.Parameter(torch.randn(K//2))

        # default L<->R swap for mirror (leave ND)
        self.emission_perm = torch.tensor([0, 2, 1])

        # pre-build mirror index
        self.mirror_idx = build_mirror_idx(self.z_dim, device=self.logpi_half.device)

        # storage
        self.logK = None

    # -------- symmetric emissions & initial (build-only; no in-place on Params) --------
    def _build_symmetric_logP_logpi(self):
        device = self.logpi_half.device
        m      = self.mirror_idx.to(device)
        K      = self.z_dim
        H      = K // 2
        L, R, ND = 1, 2, 0 
        # Indices for the mirrored half
        idx_first  = torch.arange(H, device=device)
        idx_mirror = m[idx_first]

        # logpi: tie halves equal
        logpi = torch.empty(K, device=device)
        logpi[idx_first]  = self.logpi_half
        logpi[idx_mirror] = self.logpi_half  # exact tying

        # logP: tie halves with L↔R swap
        logP = torch.empty(K, 3, device=device)
        P1 = self.logP_half
        logP[idx_first,  L]  = P1[:, L]
        logP[idx_first,  R]  = P1[:, R]
        logP[idx_first,  ND] = P1[:, ND]
        logP[idx_mirror, L]  = P1[:, R]      # swap L/R
        logP[idx_mirror, R]  = P1[:, L]
        logP[idx_mirror, ND] = P1[:, ND]
        return logpi, logP

    # -------- construct transition logits bank with symmetry by design --------
    def construct_logK(self):
        device = self.logpi_half.device
        K = self.z_dim
        S = self.S
        m = self.mirror_idx.to(device)

        self.logK = torch.zeros(S + 1, K, K, device=device)

        # Build base context matrices for action=0 contexts
        s_ids  = torch.arange(S, device=device, dtype=torch.long)
        a_bits = _bit(s_ids, self.action_bit)
        r_bits = _bit(s_ids, self.reward_bit).float().view(S, 1, 1)
        q_bits = _bit(s_ids, self.state_bit).float().view(S, 1, 1)  # 'state' bit

        idx_a0 = (a_bits == 0).nonzero(as_tuple=False).squeeze(-1)
        idx_a1 = (a_bits == 1).nonzero(as_tuple=False).squeeze(-1)

        if self.model_type in {"additive", "interaction"}:
            B  = _row_center(self.logK_base)
            Rm = _row_center(self.logK_R)
            Sm = _row_center(self.logK_S)
            if self.model_type == "interaction":
                RSm = _row_center(self.logK_RS)

            # logits(s) for action=0
            r0 = r_bits[idx_a0]   # (n_a0,1,1)
            q0 = q_bits[idx_a0]   # (n_a0,1,1)
            base = B.unsqueeze(0) # (1,K,K)

            if self.model_type == "additive":
                logK_a0 = base + r0*Rm + q0*Sm
            else:
                rq0 = r0 * q0
                logK_a0 = base + r0*Rm + q0*Sm + rq0*RSm

            self.logK[idx_a0] = logK_a0

        else:  # perstate
            # Map (r,q) -> {0,1,2,3}: idx = r*2 + q
            ctx_idx = (_bit(s_ids, self.reward_bit)*2 + _bit(s_ids, self.state_bit)).long()
            # row-center each context matrix
            ctx = _row_center(self.logK_ctx)  # (4,K,K)
            self.logK[idx_a0] = ctx[ctx_idx[idx_a0]]

        # Now fill action=1 by mirroring their a=0 counterpart
        # For any s1 with a=1, let s0 = s1 ^ (1<<action_bit)
        if idx_a1.numel() > 0:
            s1 = s_ids[idx_a1]
            s0 = s1 ^ (1 << self.action_bit)
            # gather already-built action=0 logits
            A0 = self.logK[s0]  # (n_a1,K,K)
            # mirror rows and cols
            A1 = A0.index_select(1, m).index_select(2, m)  # careful with dims
            # our self.logK is (batch, K_to, K_from). index_select(1, m): rows; (2, m): cols
            self.logK[idx_a1] = A1

        # Grounding: symmetrise by construction
        G0 = _row_center(self.logK_grounding)
        Gm = G0.index_select(0, m).index_select(1, m)
        self.logK[-1] = 0.5 * (G0 + Gm)

    # -------- assemble probabilities (no post-hoc projection needed) --------
    def get_params(self):
        # transitions
        self.construct_logK()

        # emissions & initial from symmetric half-params
        logpi_sym, logP_sym = self._build_symmetric_logP_logpi()

        # to probabilities
        self.K  = F.softmax(self.logK,   dim=-2)
        self.P  = F.softmax(logP_sym,    dim=-1)
        self.pi = F.softmax(logpi_sym,   dim=-1)

        self.altered = False
        return self.K, self.P

class HMMModel_indexK(HMMModel):
    """
    the indexK version: maximum flexibility (though questionable data effficiency): fit as many transition matrices as there are stimulus types
    """

class HMMModel_neuralK(HMMModel):
    """
    the neuralK version: uses a neural network to learn the transition matrix given stimulus types
    """




def _soft_counts_gamma(log_gamma, mask):
    """Return sum_t γ_t(i)  with padding ignored."""
    gamma = log_gamma.exp()                                # (B,T,K)
    return (gamma * mask.unsqueeze(-1)).sum(dim=(0, 1))    # (K,)

def _soft_counts_emit(log_gamma, x, mask, n_sym):
    """Return N_{i,x} = Σ_t γ_t(i)·𝟙[x_t = x]."""
    B, T, K = log_gamma.shape
    gamma = log_gamma.exp()
    out = gamma.new_zeros(K, n_sym)
    for sym in range(n_sym):
        mask_sym = (x == sym) & mask                       # (B,T)
        out[:, sym] = (gamma * mask_sym.unsqueeze(-1)).sum(dim=(0,1))
    return out                                             # (K, n_sym)

def _soft_counts_xi(log_xi, s, mask, S):
    """
    N_{from,to,stim} = Σ_t ξ_{t-1}(i,j)·𝟙[s_t = stim]
    log_xi  : (B,T-1,K,K)
    """
    B, Tm1, K, _ = log_xi.shape
    xi = log_xi.exp()
    out = xi.new_zeros(S, K, K)                            # stim × K × K
    for stim in range(S):
        mask_stim = ((s[:, 1:] == stim) & mask[:, 1:]).unsqueeze(-1).unsqueeze(-1)
        # mask_stim = ((s[:, :-1] == stim) & mask[:, :-1]).unsqueeze(-1).unsqueeze(-1)
        out[stim] = (xi * mask_stim).sum(dim=(0,1))
    return out 

def build_mirror_idx(N: int, device='cuda'):
    assert N % 2 == 0, "N must be even for symmetry"
    return (torch.arange(N, device = device) + N // 2) % N

def _row_center(M: torch.Tensor, dim=-2):
    return M - M.mean(dim=dim, keepdim=True)

def _bit(s: torch.Tensor, b: int):
    return (s >> b) & 1    