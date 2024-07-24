Goal: capture RT data spanning multiple trials with connectionist models. 
Development: Task-DyVA
- modelling sequential RT data with dynamical VAE models. 
- Constrain models to reconstruct sequences of *RT* fomr a single human participant. 

Task-switching as the main thing explored: mainly to reproduce the behavioural observation of the *switch-cost*. 

Framework: 
$$
\begin{align}
\mathbf{z}_{t + 1}  & = f_{\theta}(\mathbf{z}_t, \mathbf{u}_t, \mathbf{w}_tA)\\
\mathbf{x}_t  & \sim p_{\theta}(\mathbf{x}_t | \mathbf{z}_t)\\
\end{align}
$$
- the dynanmical system task the task stim $\mathbf{u}_t$ and noise term $\mathbf{w}_t$. 
- the dynamics function is trained to be highly expressive, and model outputs $\mathbf{x}_t$ are parametrised by a multilayer perceptron to read out from the dynamical state. The output of the network *in each channel* is a probability that the model generates a response in that direction. 

Train: 
- require an error term that measures the model's closeness to participant responses ($\delta RT \sim \mathcal{N}(0, 50\mathrm{ms})$; approximate inference framework of VAE to train. 

Model implementation: 
- Deep Variational Bayes Filter (DVBF) with a very different encoder model.

Generative model
- let latent state variables have locally linear transition dynamics. 
$$
\begin{align}
\mathbf{z}_{t + 1} & = \mathbf{A}_t \mathbf{z}_t + \mathbf{B}_t \mathbf{u}_t + \mathbf{C}_t\mathbf{w}_t\\
\mathbf{z}_{0} & = f_{\theta_{0}}(\mathbf{w}_{0})
\end{align}
$$

- $f_{\theta_{0}}$ is a MLP: {H(64, ReLU), L(16)}
- additional parameter: prior variance $\sigma_w^{2}$ to randomise the initial hidden state ($C = diag(\sigma^{2}_{w})$). 
- during training, $\mathbf{w}_t$ are sampled from the encoder model. 

- $\mathbf{A}, \mathbf{B}, \mathbf{C}$ are time-varying and all depends on matrix $\boldsymbol{\alpha}$ (can be evaluated at $\alpha_{t}^{(i)}$)
$$
\begin{align}
\mathbf{A}_t = \sum_{i = 1}^M \alpha_t^{(i)}\mathbf{A}^{(i)} \dots\\
\end{align}
$$
- $\alpha_t^{(i)}$ are determined by a single-layer nueral network $f_{\theta_{\alpha}}(\mathbf{z}_t, \mathbf{u}_t)$, softmax output. 

model output: 
$$
p_{\theta_x}(\mathbf{x}_t|\mathbf{z}_t) = \mathcal{N}(\mu_{\theta_x}(\mathbf{z}_t); 0, \mathrm{diag}(\boldsymbol{\sigma}_x^{2}))
$$
- $\mu_{\theta_x}$ is a MLP: {ReLU(64), $\sigma$(4)}
- For analysis: only use $\mu_{\theta_x}(\mathbf{z}_t)$


Encoder model: 
$$
\begin{align}
q_{\phi}(\mathbf{w}_{1:T}|\mathbf{x}_{1:T}, \mathbf{u}_{1:T}) & = q_{\phi_0}(\mathbf{w}_0)\prod_{t = 1}^{T}q_{\phi}(\mathbf{w}_t|\mathbf{x}_{t:T}, \mathbf{u}_{1:T}, \mathbf{w}_{0:T-1})\\
\end{align}
$$
breakdown: 
- we've seen the players before: x is computed from the hidden states, u is the input, and w is the variable in interest (given the noise term, everything else is deterministic). 
- backward RNN to parametrise $q_{\phi_{w}}$. 
	- state variable $\mathbf{h}_t$
	- forward pass: $\mathbf{h}_t$ and $\mathbf{z}_t$ are concatenated and passed through a MLP to get parameters of the Gaussian distributions ($\boldsymbol{\mu}_\phi$ and $\sigma_\phi$) to sample $\mathbf{w}_t$
	- same dynamics of $\mathbf{z}_t$ except that $\mathbf{w}_t$ is sampled from the encoder model.
$$
\begin{align}
\mathbf{h}_t  & = e_{\mathbf{h}_i}(\mathbf{h}_{t + 1}, e_{\mathbf{x}\mathbf{u} }(\mathbf{x}_t, \mathbf{u}_t)),  & t > 1 \\
[\mathbf{u}_\phi(\mathbf{h}_t, \mathbf{z}_t), \boldsymbol{\sigma}_{\phi}(\mathbf{h}_t, \mathbf{z}_t)]  & = e_{\mathbf{w}}(\mathbf{h}_t, \mathbf{z}_t),  & t > 1\\
q_\phi(\mathbf{w}_t|\mathbf{h}_t, \mathbf{z}_t )  & = \mathcal{N}(\mathbf{w}_t; \boldsymbol{\mu}_{\phi}(\mathbf{h}_t, \mathbf{z}_t), \mathrm{diag}(\boldsymbol{\sigma}_{\phi}^{2}(\mathbf{h}_t, \mathbf{z}_t))), & t > 1\\
\mathbf{z}_{t + 1}  & = A_{t}\mathbf{z}_t + B_t\mathbf{u}_t + C_t\mathbf{w}_t, & t > 1\\
\end{align}
$$
Netwowks: 
$$
\begin{align}
e_{\mathbf{h}_t}  & = \mathrm{LSTM}(1, 64)\\
\end{align}
$$