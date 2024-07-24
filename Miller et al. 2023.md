# Abstract
- "Disentangled RNN" for learning cognitive models.
	- Penalise for carrying information forward
# 1. Intro
Cognitive models: "software instantiations" of cognitive hypotheses --> quantitative predictions about behavioural/neural data. 
- Traditional cognitive model discovery: 
	- Human researcher, iterate {
		- propose candidate structure
		- optimise model params
		- check if model reproduces data features
	 }
	- alternative approach: fit RNNs directly to behavioural data
		- RNNs as highly expressive function approximators
		- Drawback: black box, further analyses for cognitive insights. 
- Proposed: "disentangled RNNs" for cognitive modelling
	- penalise for carrying information forward
	- interpretable, cognitive insights

"Disentangling" = learning representations in which *each dimension corresponds to a single true factor of variation*. 
- separate update rule for each element of the latent space into sub-networks
- information bottlenecks

Tasks: 
- two-arm bandit (synthetic datasets with Q-learning/Actor-Critic)
- pulse accumulation DM task

