chop Bartolo dataset into 5s clips
- $z_{0}$ 


- treat trials as our sequences
	- hence can do something similar to theirs, except we concatenate $z_{0}$ across time... 
	- $z_{0, n} = z_{T, n-1}$

1. train RNN on the task (5s clips)
	- rnn vs. gru (vs lstm)
	