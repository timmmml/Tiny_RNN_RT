import numpy as np
import joblib
import json
from .BaseTwoStepDataset import BaseTwoStepDataset, _combine_data_dict
class SimAgentDataset(BaseTwoStepDataset):
    """A dataset class suitable for all tasks in BaseTwoStepDataset.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * states * rewards)?
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=True):
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        file_path = self.data_path
        agent_path = behav_data_spec['agent_path']
        if isinstance(agent_path, str):
            agent_path = [agent_path]
        for k in agent_path:
            file_path /= k
        agent_name = behav_data_spec['agent_name']
        behav = self.behav = joblib.load(file_path / (agent_name+'.pkl'))

        with open(file_path / (agent_name+'.json'), 'r') as f:
            config = json.load(f)

        behav['trial_type'] = []
        for i in range(len(behav['action'])):
            if config['task'] == 'Akam_RTS':
                behav['trial_type'].append(behav['action'][i] * 4 + behav['stage2'][i] * 2 + behav['reward'][i])
                self.unique_trial_type = 8
            else:
                raise NotImplementedError
        if verbose: print('Total trial num:', self.total_trial_num)
