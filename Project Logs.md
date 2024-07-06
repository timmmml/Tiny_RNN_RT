# Project logs
## Things to try out before initial meeting with Marcelo

- [x] Implement a first batch of training
	- [x] step 1: map from decision variable (currently recorded as scores in `RTS_RL_agents_v1\core.py`, used as a model for the `agent`. The two structures are both used for simulation.)
		- [x] detailed approach: design an orthogonal class called Behav2Dataset, which reads out from an agent simulation structure (currently a `json` in the SIM_DATA_PATH), outputs the appropriate training label per `training_id` to be specified in the configs. 
		- [x] parallel to the above class create a ddm-linker to generate simulatory reaction times.
		- [ ] have an umbrella dictionary for mapping agent ids to ddm parameters
### Model specification

Let's start with specifying model Task 1.1. (corresponding to the RTS task in Ji-an's paper)
- **inputs**: four modalities of time series, packaged finally as torch.Tensor((batch_size, sequence_length, 4))
	- **first modality**: go-cue (1 for go, 0 for no go)
	- **second modality**: past trial's second-stage state ($\pm 1$, 0 for no second-stage state)
	- **third modality**: past trial's reward ($\pm 1$ for positive/negative reward; 0 for no reward received))
	- **fourth modality**: past trial's action ($\pm 1$ for positive/negative choice; 0 for no choice produced)
- *note here we don't **need** to include the model's choice as input, but we may try doing that as a teacher's forcing technique* 
- *all modalities are presented on go; then silence for 10 timesteps (hence 300ms per step)*

- **outputs**: one or two modalities
	- case 1: **one modality**, using the sign to encode the choice (positive/negative) and magnitude to include probability of choice at that time. 
	- case 2: **two modalities**
		- case 2.1: one for choice A and one for choice B. Note, these will all be probabilities. They don't have to add up to 1 (the other choice is not making a decision right then. we will need to *think about how we use a differentiable loss function to grade it*)
		- **case 2.2**: one for acting at that time (probability), and one for the choice (binary). 
			*perhaps it may be simpler to construct a loss function here?*
	- total loss: 
		- **decision + reaction time loss**: a kind of combined loss function that is lowest for the correct decision point-on with the correct reaction time. 

Discussion on the loss function:

| decision correct | reaction time distribution | loss   |
| ---------------- | -------------------------- | ------ |
| Yes              | spot on                    | 0      |
| Yes (when?)      | high divergence            | low    |
| No (when?)       | low divergence             | medium |
| No               | high divergence            | high   |
Main consideration:
 - how to answer the "when" questions? we can implement two loss functions, one to grade the train of decisions, in the sense that every such decision contributes some weight to the final loss function, and the weight depends on the reaction probability (maybe this is the loss to use! think about how we can formalise it)
	 - choice A: combine this loss with KL divergence-type loss as a chimerical loss function (hard to think about how)
	 - **choice B**: use this loss (let's say we apply a Gaussian Kernel to have a nonzero value label per bin, and then compute a distance (quadratic) to the signed prediction) + KL divergence-type loss.

- do we want regularisation loss as well?

- [ ] step 2: train tasks (make sure I have 1.1 done before meeting with Marcelo. Can do an initial attempt at 2)
	- [ ] Task 1.1: Simple RNN to predict not only decision label but also reaction time. Experiment with a few dimensions. 
	- [ ] Task 1.2: decision label and reaction time put into a custom loss function, encouraging the network to make a decision at the right time and with the right label. In effect, let's credit having the correct label *around* the time (so we weigh the thing). (Should write this thing in torch.tensor, vectorised, for most efficient training; let's think about Gaussian weighting of decision importance (closeness to the response time)). 

### Notes for Marcelo meeting 
#### Before: 

- questions: Why tiny RNNs (not VAEs or other distribution models)? 
	- Li et al. 2023: tiny RNNs situate between large-scale machine learning models (such as higher-order RNN structures) and lower-parameter statistical models (interpretable parameters such as model-based RL models). 
		- 2/3 units are sometimes indeed better than larger-scale models in cognitive task modelling *(check if this is going to be the case for reaction times)* *(check additional model training considerations that may have hindered larger-scale networks)*.
		- Dynamical system analyses (phase portraits based on plotting dynamical system response to changes - lower unit number = lower dimensionality)·
	**Q1**: inherently low-dimensioned systems vs. PCA dimensionality reduced higher-ordered systems (phase portraits vs. other latent dynamics analyses)

**Q2**: *project related* - are we trying to perform better than DDM? For our purposes, I think DDM has a very nice feature, in that it gives a lot of flexibility in our data (in genearting a probability distribution of per-trial reaction times). 
- in practice (e.g. for behavioural datasets, we could use distributional studies tools to do essentially the same thing (should we implement Task-DyVA instead of competing with it?))
	- we should think about where we are placed within the literature: as a rival to the DDM/Task-DyVA, or as a down-the-stage dynamics analysis method. 

- we can compare lean versions with say Gaussian loss functions on the predicted dynamics against those that lean more onto fitted RT predictions as annotation to the dataset.

**Q3**: *project related* dealing with undecision trials.

Open-ended. 
*predictability vs. interpretaibility*
- complicated machine learning models to even predict neural activity
	- not assuming anything about the model approximating the brain. 
- we try to make the structure similar to the brain
	- approximate the way a decision is made. 

#### After: 

### Notes for self

- **Difficulty**
	- Typical models are across trials (RNNs reflect input-output structures across trials). If we want to train an RNN that not only captures the overall computation but the within-trial dynamics, the RNN must also mimic the unrolling of the decision variable within a trial. 
		- Let's say, we may be able to do this by presenting the trial as a sequence of inputs, externally model a process in which more information is being accumulated to the network, or make the network behave in absence. 
		- For better training, we may use an additional behavioural control unit that is specialised to gate behaviour from decision. Specifically, the unit operates on the information available to show when to emit the decision (whereas say another unit is responsible for the decision itself). Can define a combined loss function. 
			- Can compare this approach with one-unit-handle-all situation. 

- **Explicit alignment with evidence accumulation**
	- Can think about how to use structured RNNs with evidence-tracking units.
	- For example, a threshold-crossing unit to gate decision variable and hence emitted decision, within trial, across RNN dynamics. 


- RL-DDM
	- RL=DDM i
- Tiny-RNN = RL
- RT-RNN = RL-DDM (drop DDM assumption and learn directly from data)
- goal: trial-by-trial, individual participant, choice and RT prediction
	- use both information to fit models
	- we hope the model provides more information than choice alone
	- (maybe it's plausible to do only-choice, for the scarcity of the data)
- one question: does RT help/hurt the predictions
- in what conditions do they provide mechanistic insights. 

Simple way: Jaffe et al. --> RL task; interpretable. 

Data: Botolo et al. (Ramon Bartolo and Bruno B Averbeck. Prefrontal cortex predicts state switches during reversal learning. Neuron, 106 (6):1044–1054, 2020.)
monkey data.
- read the paper, formulate hypothesis on variability in reaction time (reversal learning task)
	- formalise hypotheses (higher RT after what?)
	- ultimately we want the model to capture the variabilty in RT. 
- look at the distribution of reaction times
- core part: augment from Jaffe et al. Ji-an's work
- implement Jaffe et al. on Bartolo

David Sussillo 
Larry Abbott
- strategy to fit RNN on spike data. 
- GRU/LSTM: good for getting around gradient issues - (not much used for continuous modelling tasks - may be possible to find that GRU is )

How are units changing activities given input? 

for large models, use PCA
- but PCA *doesnt work* --> no good solution
- JI-AN: let's use small unit sizes (good and orthogonal)


Victory: 
- prediction and interpretabilty
- Jaffe et al. (or ML methods): interpretability is 

Predictability (of RT) at least as good at other models (Jaffe and RTNet, and RL-DDMs)
Interpretability: reconstruct insights from say RL-DDM, rediscover the same insights and more. 

Competitors to beat: 
- Jaffe et al. and DDM. 