"""This is my hand-implemented Task-DyVA model from Jaffe et al.

Some params still to be tweaked by the user (me), conferred in the configs dictionary.

NOTE: this is not used. 
"""
import torch
import torch.nn as nn
from .BaseNNAgent import  BaseNNAgent
from torch.distributions import MultivariateNormal as Normal

class TaskDyVA(BaseNNAgent):
    """Docstring to be added"""
    def __init__(self, model_params):
        super(TaskDyVA, self).__init__()

        self.batch_size = model_params.get('batch_size', 32)

        # x, z, u, w dimensions (action modality space, latent space, input space, and stochastic variable space)
        self.x_dim = model_params.get('x_dim', 2)
        self.z_dim = model_params.get('z_dim', 16)
        self.u_dim = model_params.get("u_dim", 4)
        self.w_dim = model_params.get('w_dim', 16)
        self.h_dim = model_params.get('h_dim', 64)

        # number of basis matrices to cook into A_t, B_t, C_t
        self.M = model_params.get("basis_num", 2)

        # f_{theta_0} layer init: 2 Layers; w_size -> hidden_size -> z_size
        self.theta_0_hidden_size = model_params.get('theta_0_hidden_size', 64)
        self.theta_0_hidden_actvation = model_params.get('theta_0_hidden_activation', 'relu')
        self.theta_0_output_size = self.z_dim
        self.theta_0_output_activation = model_params.get('theta_0_output_activation', 'linear')

        self.f_theta_0 = nn.Sequential(
            nn.Linear(self.w_dim, self.theta_0_hidden_size),
            self._activation_fn(self.theta_0_hidden_actvation),
            nn.Linear(self.theta_0_hidden_size, self.theta_0_output_size)
        )

        # set up the basis matrices for the generative model
        # these are used to premultiply z_t, u_t, and w_t, specifying the dynamics of the system.
        self.A = nn.Parameter(torch.randn(self.M, self.z_dim, self.z_dim))
        self.B = nn.Parameter(torch.randn(self.M, self.z_dim, self.u_dim))
        self.C = nn.Parameter(torch.randn(self.M, self.z_dim, self.w_dim))

        self.sigma_w_p = nn.Parameter(torch.eye(self.w_dim))
        self.mu_w_p = nn.Parameter(torch.zeros((self.w_dim)))
        self.mu_w_p.requires_grad = False


        # Note: f_{theta_alpha} is used to linearly shape the basis matrices A, B, and C into A_t, B_t, and C_t
        # f_{theta_alpha} layer init: 1 Layer:  z_size + w_size -> alpha_size (M)
        self.f_theta_alpha = nn.Sequential(
            nn.Linear(self.z_dim + self.u_dim, self.M),
            nn.Softmax(dim=-1)
        )

        # mu_theta_x layer init:
        self.theta_x_hidden_size = model_params.get('theta_x_hidden_size', 64)
        self.theta_x_hidden_activation = model_params.get('theta_x_hidden_activation', 'relu')
        self.theta_x_output_size = self.x_dim
        self.theta_x_output_activation = model_params.get('theta_x_output_activation', 'linear')

        self.mu_theta_x = nn.Sequential(
            nn.Linear(self.z_dim, self.theta_x_hidden_size),
            self._activation_fn(self.theta_x_hidden_activation),
            nn.Linear(self.theta_x_hidden_size, self.theta_x_output_size)
        )

        # currently sigma_x is a constant diagonal of 0.75s
        self.sigma_x_scalar = model_params.get('sigma_x', 0.75)
        self.sigma_x = nn.Parameter(torch.eye(self.x_dim) * self.sigma_x_scalar)
        self.sigma_x.requires_grad = False

        # set up the encoder components to get a distribution of w given x and u
        self.e_ht_hidden_size = model_params.get('e_ht_hidden_size', 64)
        self.e_xu_hidden_size = model_params.get('e_xu_hidden_size', 64)
        self.e_xu_activation = model_params.get('e_xu_activation', 'relu')
        self.e_xu_output_size = self.h_dim

        self.e_ht = nn.LSTM(
            input_size= self.e_xu_output_size,
            hidden_size=self.e_ht_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.e_xu = nn.Sequential(
            nn.Linear(self.x_dim + self.u_dim, self.e_xu_hidden_size),
            self._activation_fn(self.e_xu_activation),
            nn.Linear(self.e_xu_hidden_size, self.e_xu_output_size)
        )

        self.e_w_hidden_size = model_params.get('e_w_hidden_size', 64)

        self.e_w = EW(self.h_dim + self.z_dim, self.e_w_hidden_size, self.w_dim)

        self.reconstruction = torch.zeros(1)
        self.kl = torch.zeros(1)

    def forward(self, x, u):
        # use the generative model to decode the latent state z
        # x is of shape (batch_size, seq_len, x_dim)
        # u is of shape (batch_size, seq_len, z_dim)
        # w is sampled dynamically
        print(x.shape, u.shape)

        self.sigma_w = self.sigma_w_p.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.mu_w = self.mu_w_p.unsqueeze(0).repeat(self.batch_size, 1)
        w0 = self.sample_reparametrize(self.mu_w, self.sigma_w)
        self.kl = Normal(self.mu_w, self.sigma_w).log_prob(w0).sum()

        z1 = self.f_theta_0(w0)
        z = torch.zeros((u.shape[0], u.shape[1], self.z_dim))
        w = torch.zeros((u.shape[0], u.shape[1], self.w_dim))
        # NOTE: w0 is not used in the big w tensor. This is because w0 is only used to get z1.

        z_t = z1
        w_t = w0

        xu = torch.cat([x, u], dim=-1)
        xu = self.e_xu(xu)  # xu is of shape (batch_size, seq_len, e_xu_output_size).
        xu = torch.flip(xu, dims=[1])
        h, _ = self.e_ht(xu)  # h is of shape (batch_size, seq_len, h_dim)
        h = torch.flip(h, dims=[1])

        for t in range(0, u.shape[1]):
            alpha_t = self.f_theta_alpha(torch.cat([z_t, u[:, t, :]], dim=1))
            # alpha_t is of shape (batch_size, M), A is of shape (M, z_dim, z_dim)
            # use the encoder to get a distribution of w given x and u

            mu_w, sigma_w = self.e_w(torch.cat([h[:, t, :], z_t], dim=-1))
            print(t)
            sigma_w = torch.diag_embed(sigma_w**2 * 16 + 1e-6)
            w_t = self.sample_reparametrize(mu_w, sigma_w)
            self.kl += Normal(mu_w, sigma_w).log_prob(w_t).sum() # conditional
            self.reconstruction += Normal(self.mu_w, self.sigma_w).log_prob(w_t).sum() #prior
            w[:, t, :] = w_t
            z[:, t, :] = z_t
            z_t = torch.einsum('ijk,ik->ij', torch.einsum('ij,jkm->ikm', alpha_t, self.A), z_t) + torch.einsum('ijk,ik -> ij', torch.einsum('ij,jkm->ikm', alpha_t, self.B), u[:, t, :]) + torch.einsum('ijk, ik -> ij', torch.einsum('ij,jkm->ikm', alpha_t, self.C), w_t)

        mu_x, sigma_x = self.mu_theta_x(z), self.sigma_x
        self.reconstruction = Normal(mu_x, sigma_x).log_prob(x).sum().sum()
        return mu_x, sigma_x

    def to(self, device):
        self.device = device
        self.reconstruction = self.reconstruction.to(device)
        self.kl = self.kl.to(device)
        return super().to(device)

    def sample_reparametrize(self, mu, sigma):
        """Reparametrize the distribution"""
        eps = torch.randn_like(mu, device=mu.device).unsqueeze(-1)
        return mu + torch.einsum("ijk,ijk->ij", sigma, eps)

    def elbo_loss(self, c_i):
        """Calculate the ELBO loss

        Args:
                c_i: scalar annealing factor
        """
        return -self.reconstruction * c_i + self.kl

    def _activation_fn(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        else:
            raise ValueError(f"Activation function {activation} not recognized.")

class EW(nn.Module):
    """Implements the custom E_w network"""
    def __init__(self, input_size, hidden_size, output_size):
        super(EW, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.mu_dist = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        self.sigma_dist = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """Forward pass of the E_w network
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size = h_dim + z_dim)"""
        h = self.hidden(x)
        mu = self.mu_dist(h)
        sigma = self.sigma_dist(h)
        return mu, sigma


