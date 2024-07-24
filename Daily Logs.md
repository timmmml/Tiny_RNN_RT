This file is for logs only. See [[Project Logs]] for more detailed notes. 

# 07/06/2024, 09/07/2024
Recap from Marcelo's meeting, we agreed that the first steps is two experiment with Task-DyVA implementation on the Bartolo dataset. 

At this stage, there are two things to be done. 
1. [x] implement the Task-DyVA by its specs from the paper. 
	- [ ] re-map the trainer class for Task-DyVA. 
	- [ ] debug the initial implementation with simulated data
2. [ ] find the Bartolo dataset and do a batch of initial experimentation/exploration. 

Choice: 
- in Jaffe et al. 2023, they chopped up each 60 second gameplay into short 5 seconds segements and amplified the data by synthetic data points.
	- Specifically, they *estimated RT pdfs corresponding to 4 types of trials* (congruent stay, congruent switch, incongruent stay, incongruent switch). 
	- First introduce 10 folds replication into the original dataset
	- Then mutate it $75\%$ of times. 
		- Sample a trial type $c$
		- Sample trial condition (including stimulus)
		- Sample RT from $P(\mathrm{RT}|c)$, which is a Gaussian Kernel Estimation of the empirical RTs.
		- Set choice to the correct one. 
		- repeat until reaching 5s
- stitch them together for a continuous time series. 
	
- This is less reasonable in our case
Let's start with the first one for it helps to first understand model in a functional manner.