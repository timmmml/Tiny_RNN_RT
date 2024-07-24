# Outline

1. Compare and contrast RTNet vs. Task-DyVA. 
2. Discuss specific implementation of Task-DyVA. 
3. Bartolo dataset brief words

# RTNet vs. Task-DyVA

## RTNet idea

**Uses a stochastic network (BNN) to imitate evidence accumulation directly from image sources (drawn digits)**

Task used: perceptual decision making (drawn digit inference under varying levels of noise)

- RT phenomenon: speed-accuracy tradeoff + *Stochasticity* (measured as consistency across evaluations)

### Detailed Implementations

Competitors:
- CNet (ResNet18 architecture) = *Parallel Cascaded Network - residual network structure with propagation delays: **stable activation** achieved in later residual blocks only when the full length is traveled, else **partial activation**. Decision can happen on any step $t$, given a readout layer; if $t < N_{\mathrm{res}}$ this decision is based on partial input in later blocks*
![[CNet illustration.png]]

- MSDNet (AlexNet architecture; 5 + 3) = *Multi-scale dense netowrk - single input layer + $n$ hidden layers, each with its own output layer. Hence MSDNet can make a separate decision on any layer - early decoding if any one layer decides there is evidence enough to decide.*
![[MSDNet.png]]

In a nutshell: RTNet (AlexNet architecture) = *CNN but with sampled weights; a stimulus is present multiple times through the network, each time with a different weights. **Accumulate evidence** through processing steps, decision on threshold. Hence **stochastic decision and variable RT***
![[RTnET.png]]
Details: 
- BNN to generate noisy weights (fitted instead of weight values, distributions)

Notably, CNet and MSDNet build reaction-time into the network structure and decides reaction time in a deterministic manner given input. RTNet has built-in stochasticity. 

Drawback: no recurrence. 

Inspiring aspect: sampling based inference; *race model* of decision making: 
- noisy accumulator process for each choice. 

Tests for the netowrk: 
- task = hand-written image with manipulated noise levels as well as manipulated speed-accuracy tradeoff (asking the subjects to focus on one or the other)
- each image presented twice for measuring stochasticity
- CNet, MSDNet, RTNet challenged with the same image set
	- Caveat: *images fed are not the same ones actually; readjusted stim noise to match overall accuracy to that produced by humans in each case*
	- for speed-accuracy tradeoff within the networks, *adjust threshold to meet human accuracy on speed/accuracy conditions*. 
- Trained $N = 60$ of each models, each with its own initial params (fitted signal strength) before training began, in order to match $N = 60$ human subjects. 

Results: 
- all of these networks demonstrate speed-accuracy tradeoff, though only RTNet shows stochasticity (by design). 
- RTNet captures skewness better. 
- et cetera on speed-accuracy related predictions and results

Notes on biological plausibility: 
- the authors offered an account on how one forward sweep from retina to IT being $70-100ms$ but human RT can be hundreds of $ms$ to second-timescale. 
	- signature of recurrence (top-down connections even in the visual stream, similar to a message passing manner (re-information theory decoder models))
- MSDNet - decision after each layer, not the case in the brain
- CNet is closer - parallel and continuous processing, delayerd transmission inspired by biololgy in the first place: delay across cortical layers. but no stochasticity nor recurrence. 
- RTNet captures *noisiness (repeated presentation, noisy weights), multiple-passing (multiple ff sweeps in time scale), accumulation process*. Crucially lacks recurrence, still, nor nonlinear interaction between passes (only accumulation and addition). 

Methodological considerations
- different ways to generate noise in the weights; 
	- same amount of variability to all weights; too small for some, too large for others. no gain from evidence accumulation, or detrimental effects. 
	- hence Bayesian nueral network that yields posterior distributions for each weight (then can generate appropriate noise for each weight). 
- (noisy weights -> noisy activations) vs. noisy activations only? 
	- they used random weights just because BNNs are ready-made. 
	- these are effectively the same, argued authors. 

Limitation (as previously mentioned)
1. non-optimal accumulation process (threshold crossing; needs to basically guess between close competitors) 
2. each ff sweep is independent; no nonlinear interaction between passes. This can be modified for dependency such as by conditioning the current weight posterior on the previous timesteps. (previous timestep weights (authors mentioned); or even previous value output acitvations (my guess - this way we bring true recurrence))

****
## Task-DyVA idea

**Uses an expressive dynamical system to do variational inference on the multivariate distribution of task rollout as time series**

Task used: task-switching - Ebb and Flow. Pointing vs. motion as the target, color to cue the task identity.

- RT phenomenon: coherent vs. incoherent trials; task-switch costs. 

### Detailed implementation

![[Factor graph TDV.png]]

Goal: $p(\mathbf{x}_{1:T}, \mathbf{w}_{1:T}|\mathbf{u}_{1:T})$ and $q_{\phi}w_{1:T} |\mathbf{x}_{1:T}, \mathbf{u}_{1:T}$

**Generative model**: 
$$
\begin{align}
\mathbf{z}_{t+1} & = f_{\theta_{z}}(\mathbf{z}_{t}, \mathbf{u}_{t}\mathbf{w}_{t})\\
& = \mathbf{A}_{t} \mathbf{z}_{t} + \mathbf{B}_{t}\mathbf{u}_{t} + \mathbf{C}_{t}\mathbf{w}_{t} & t > 1\\
\mathbf{z}_{t}  & = f_{\theta_{0}}(\mathbf{w}_{0})& t = 0 \\
f_{\theta_{z}}  & = MLP(64\times\mathrm{ReLU},16\times\mathrm{L}) \\

 \\
\mathbf{w}_{t}  & \sim \mathcal{N}(0, \mathrm{diag}(sigma_{w}^{2}))  & \text{for task simulation}\\
\mathbf{w}_{1:T}  & \sim q_{\phi}(\mathbf{w}_{1:T}|\mathbf{u}_{_1:T}, \mathbf{x}_{1:T}) & \text{during training} \\
 \\
define &: \{ \mathbf{A}^{(i)}, \mathbf{B}^{(i)}, \mathbf{C}^{(i)}\}_{i = 1\dots M} \\
\boldsymbol{\alpha}_{t}  & = f_{\theta_{\alpha}}(\mathbf{z}_t, \mathbf{u}_{t}) & f_{\theta_{\alpha}} = MLP(M \times \mathrm{SoftMax}) \\
 \\
\mathbf{A}_{t}  & = \sum_{i = 1}^M \alpha_{t}^{(i)}\mathbf{A}^{(i)} \\
\mathbf{B}_{t}  & = \sum_{i = 1}^M \alpha_{t}^{(i)}\mathbf{B}^{(i)} \\
\mathbf{C}_{t}  & = \sum_{i = 1}^M \alpha_{t}^{(i)}\mathbf{C}^{(i)} \\
 \\
p_{\theta_x}(\mathbf{x}_{t}|\mathbf{z}_{t})  & = \mathcal{N}(\mathbf{x}_{t}; \mu_{\theta_{x}}(\mathbf{z}_{t}), \mathrm{diag}(\sigma_{x}^{2})) & \mu_{\theta_{x}} = MLP(64 \times \mathrm{ReLU}, 4 \times \sigma), \sigma_{x}^{2} & = 0.75
\\
\end{align}
$$

**Encoder model**: 
- goal is to model the distribution for $p(x_{1:T}|u_{1:T})$; to do this, model $p(z_{1:T}|x_{1:T}, u_{1:T})$
- as $p(z_{1:T}|x_{1:T}, u_{1:T}) = p(w_{1:T}|x_{1:T}, u_{1:T})$ from deterministic relationships
$$
\begin{align}
p_{\theta}(\mathbf{w}_{1:T}|\mathbf{u}_{1:T}, \mathbf{x}_{1:T})  & \approx q_{\phi}(\mathbf{w}_{1:T}|\mathbf{u}_{1:T}, \mathbf{x}_{1:T}) \\
 & = q_{\phi_{0} }(\mathbf{w}_{0})\prod_{t = 1}^{T}q_{\phi_{w}}(\mathbf{w}_{t}|\mathbf{w}_{0:t-1}, \mathbf{u}_{1:T}, \mathbf{x}_{1:T}) \\
  q_{\phi_{w}}(\mathbf{w}_{t}|\dots)& = q_{\phi}(\mathbf{w}_{t}|\mathbf{h}_{t}, \mathbf{z}_{t}) \\
 & = \mathcal{N}(\mathbf{w}_{t}; \mu_{\phi_{w}}(\mathbf{h}_{t}, \mathbf{z}_{t}), \mathrm{diag}(\sigma_{\phi_{w}}^{2}(\mathbf{h}_{t}, \mathbf{z}_{t})) \\ \\
 q_{\phi_{0}}(\mathbf{w}_{0})  & = \mathcal{N}(\mathbf{w}_{0}; 0, \mathrm{diag}(\sigma_{w}^{2})) , z_{1} = f_{\theta_{0}}(\mathbf{w}_{0})\\
  \\

\forall t>1: \\
\mathbf{h}_{t}  & = e_{\mathbf{h}_{t}}(\mathbf{h}_{t + 1}, e_{\mathbf{x}\mathbf{u}})(\mathbf{x}_{t}, \mathbf{u}_{t})  \\
\left[ \boldsymbol{\mu}_{\phi}(\mathbf{h}_{t}, \mathbf{z}_{t}), \boldsymbol{\sigma}_{\phi}(\mathbf{h}_{t, \mathbf{z}_{t}}) \right]  & = e_{\mathbf{w}}(\mathbf{h}_{t}, \mathbf{z}_{t})  \\
\mathbf{z}_{t}  & \text{ as above}
 \\

e_{\mathbf{h}_t}  & = LSTM(64),  \\
e_{\mathbf{x}\mathbf{u}}  & = MLP(64 \times \mathrm{ReLU}, 64 \times \mathrm{L})\\
e_{\mathbf{w}}  & = MLP(64 \times \mathrm{ReLU}, \left[ 16 \times \mathrm{L}; 16 \times \mathrm{SoftMax}\right] ) & \sigma_{\phi} = out(e_{\mathbf{w}})[2] \times 16 + 1\mathrm{e}-6


\end{align}
$$

**Objective**
- slightly modified ELBO
$$
\begin{align}
\mathcal{L}(\theta, \phi; \mathbf{x}_{1:T}, \mathbf{u}_{1:T}) 
 & = \mathbb{E}_{q_{\phi}(\mathbf{w}_{1:T}|\mathbf{x}_{1:T}, \mathbf{u}_{1:T})}[c_{i}\log p_{\theta}(\mathbf{x}_{1:T}, \mathbf{w}_{1:T}|\mathbf{u}_{1:T}) - \log q_{\phi}(\mathbf{w}_{1:T}|\mathbf{u}_{1:T}, \mathbf{x}_{1:T})] \\
  \\
 p_{\theta}(\mathbf{x}_{1:T}, \mathbf{w}_{1:T}|\mathbf{u}_{1:T})   & = \prod p_{\theta}(\mathbf{x_t}|\mathbf{z}_{t}) p_{\theta_{w}}(w_{1:T}) \\
q_{\phi}(\mathbf{w}_{1:T}|\mathbf{x}_{1:T}, \mathbf{u}_{1:T})  & \text{ as above}
\end{align}
$$

Training: 5s clips only! (think about how to do our data with this - maybe as easy as fittable initial conditions?)
## Do we have the right dataset? 

- see shared screen for some RT exploration