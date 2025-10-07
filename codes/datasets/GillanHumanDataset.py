import os
import joblib
import numpy as np
import torch
from .BaseTwoStepDataset import BaseTwoStepDataset
from path_settings import *


class GillanHumanDataset(BaseTwoStepDataset):
    """A dataset class for the two-step with binary actions and rewards.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * rewards = 4 combinations)
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=True):
        self.unique_trial_type = 16 # 2 first-stage actions * 2 rewards * 4 second-stage actions
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def _detect_trial_type(self):
        """Determine trial type from behavioral data."""

        behav = self.behav
        behav['trial_type'] = []
        for b in range(self.batch_size):
            stage_1_selected_stimulus = self.behav['stage_1_selected_stimulus'][b].copy()  # 1, 2, -1
            stage_1_selected_stimulus[stage_1_selected_stimulus == -1] = 1 # 1, 2
            stage_1_selected_stimulus -= 1 # 0, 1
            assert len(np.unique(stage_1_selected_stimulus)) == 2

            stage_2_state = self.behav['stage_2_state'][b].copy() # 2, 3, -1
            stage_2_state[stage_2_state==-1] = 2 # 2, 3
            stage_2_state -= 2 # 0, 1
            assert len(np.unique(stage_2_state)) == 2

            stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy()  # 1, 2, -1
            stage_2_selected_stimulus[stage_2_selected_stimulus == -1] = 1 # 1, 2
            stage_2_selected_stimulus -= 1 # 0, 1
            assert len(np.unique(stage_2_selected_stimulus)) == 2

            reward_masked = self.behav['reward'][b].copy()
            reward_masked[reward_masked==-1] = 0 # 0, 1
            assert len(np.unique(reward_masked)) == 2

            # trial_type = stage_1_selected_stimulus * 8 + stage_2_state * 4 + stage_2_selected_stimulus * 2 + reward_masked
            trial_type = list(zip(stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked))
            behav['trial_type'].append(trial_type)

    def _load_all_trial_type(self, behav_data_spec):
        """Create artificial dataset with all trial types."""
        pass
        # np2list = lambda L: [np.array([x]) for x in L]
        # self.behav = {
        #     'action': np2list([0, 0, 1, 1]),
        #     'stage2': np2list([0, 0, 1, 1]),
        #     'reward': np2list([0, 1, 0, 1]),
        # }
        # self._detect_trial_type()

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load behavioral and neural data.

        The loaded data is stored in self.behav, the standard format of behavioral data.

        Args:
            behav_data_spec: A dictionary of behavioral data specification.
            neuro_data_spec: A dictionary of neural data specification.
                select_bins
         """

        if neuro_data_spec is None:
            neuro_data_spec = {}
        self.behav_data_spec = behav_data_spec
        self.neuro_data_spec = neuro_data_spec

        if not os.path.exists(self.data_path / 'Gillandata_behav.pkl') or bool(input("redo?")):
            _combine_subjects_data(self.data_path / 'Gillandata_behav.pkl')
        subject_remapping, behav = joblib.load(self.data_path / 'Gillandata_behav.pkl')
        behav['action'] = behav['stage_1_selected_stimulus']
        self.subject_remapping = subject_remapping
        self.behav = behav
        self.neuro = neuro = {}

        behav['aug_block_number'] = list(np.arange(len(behav['action']))) # list of numbers, the block number to be augmented
        if 'augment' in behav_data_spec and behav_data_spec['augment']: # only used in distillation
            augment_level = behav_data_spec['augment_level'] if 'augment_level' in behav_data_spec else 1
            self._augment_data(augment_level=augment_level)
        self._detect_trial_type()
        print("===loaded all===", 'Gillan Human')
        print('Total batch size:', self.batch_size)
        print('Total trial num:', self.total_trial_num)

    def get_behav_data(self, batch_indices, format_config=None, remove_padding_trials=False, selected_trial_indices=None):
        """ overwrite the get_behav_data function in BaseTwoStepDataset with remove_padding_trials option"""
        assert not remove_padding_trials
        if self.behav_format == 'tensor' and selected_trial_indices is not None:
            # selected_trial_indices is at the trial level (in total 200)
            # for cog_session, selected_trial_indices can be directly used
            # for tensor, the time point is at the stage level (in total 400), we should convert selected_trial_indices to time point level
            selected_trial_indices = np.array(selected_trial_indices)
            selected_trial_indices = np.sort(np.concatenate([selected_trial_indices*2, selected_trial_indices*2+1]))
        return super().get_behav_data(batch_indices, format_config, remove_padding_trials, selected_trial_indices)

    def _behav_to_cog_sessions(self, format_config):
        """Transform standard behavioral format to cog_session format, stored in cog_sessions attribute.

        Args:
            format_config: A dict specifies how the standard data should be transformed.
        """
        if self.cog_sessions is not None:
            return
        self.cog_sessions = []
        behav = self.behav
        print('Transforming standard format to cog_session format...')
        for block_idx in range(self.batch_size):
            stage_1_selected_stimulus = behav['stage_1_selected_stimulus'][block_idx].copy()  # 1, 2, -1
            stage_2_state = behav['stage_2_state'][block_idx].copy()  # 2, 3, -1
            stage_2_selected_stimulus = behav['stage_2_selected_stimulus'][block_idx].copy()  # 1, 2, -1
            reward = behav['reward'][block_idx].copy()  # 0, 1, -1
            mask_1 = (stage_1_selected_stimulus != -1) * 1
            mask_2 = (stage_2_selected_stimulus != -1) * 1
            mask = np.array([mask_1, mask_2]).T # shape: trial_num, 2
            stage_1_selected_stimulus[stage_1_selected_stimulus == -1] = 1  # 1, 2
            stage_1_selected_stimulus -= 1  # 0, 1
            stage_2_state[stage_2_state == -1] = 2  # 2, 3
            stage_2_state -= 2  # 0, 1
            stage_2_selected_stimulus[stage_2_selected_stimulus == -1] = 1  # 1, 2
            stage_2_selected_stimulus -= 1  # 0, 1
            reward[reward == -1] = 0  # 0, 1
            transitions = stage_1_selected_stimulus == stage_2_state # 1 for common transition, 0 for rare transition
            trial_num = len(reward)
            self.cog_sessions.append({
                'n_trials': trial_num,
                'first_choices': stage_1_selected_stimulus,
                'second_states': stage_2_state,
                'second_choices': stage_2_selected_stimulus,
                'outcomes': reward,
                'transitions': transitions,
                'mask': mask,
            })
        print('\nTotal block num', self.batch_size)

    def _behav_to_tensor(self, format_config):
        """Transform standard behavioral format to tensor format, stored in torch_beahv_* attribute.

        standard format (list of 1d array) -> tensor format (2d array with 0 padding).
        The attributes are:
            torch_beahv_input: tensor format of agent's input
            torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
            torch_beahv_target: tensor format of agent's target output
            torch_beahv_mask: tensor format of agent's mask (1 for valid trials, 0 for padding trials)

        Not use nan padding:
            rnn model make all-nan output randomly (unexpected behavior, not sure why)
            the one_hot function cannot accept nan
            long type is required for cross entropy loss, but does not support nan value

        Args:
            format_config: A dict specifies how the standard data should be transformed.

        """
        if self.torch_beahv_input is not None:
            return
        trial_num = max([len(block) for block in self.behav['reward']])
        assert trial_num == 200
        act_num = 6
        state_num = 3
        sub = np.zeros((self.batch_size, 2*trial_num,1))
        state = np.zeros((self.batch_size, 2*trial_num, state_num))
        act = np.zeros((self.batch_size, 2*trial_num, act_num))
        rew = np.zeros((self.batch_size, 2*trial_num, 1))
        target = np.zeros((self.batch_size, 2*trial_num))
        target_mask = np.ones((self.batch_size, 2*trial_num, act_num))
        mask = np.zeros((self.batch_size, 2*trial_num))
        for b in range(self.batch_size):
            # one-hot encoding for action and state
            sub[b, :, 0] = self.behav['sub_id'][b]
            stage_2_state = self.behav['stage_2_state'][b].copy() # 2, 3, -1
            stage_2_state[stage_2_state==-1] = 2
            state[b, np.arange(0, 2*trial_num, 2), stage_2_state - 1] = 1 # second-stage state 2-> 1, state 3->2
            state[b, np.arange(1, 2*trial_num, 2), 0] = 1 # first-stage state == 0

            stage_1_selected_stimulus = self.behav['stage_1_selected_stimulus'][b].copy() # 1, 2, -1
            stage_1_selected_stimulus[stage_1_selected_stimulus==-1] = 1
            act[b, np.arange(0, 2*trial_num, 2), stage_1_selected_stimulus-1] = 1 # first-stage action 1-> 0, action 2->1
            stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy() # 1, 2, -1
            stage_2_selected_stimulus[stage_2_selected_stimulus==-1] = 1
            action_baseline = stage_2_state * 2 - 2 # 2, 4
            # stage_2_state == 2: second-stage action 1-> 2, action 2->3
            # stage_2_state == 3: second-stage action 1-> 4, action 2->5
            act[b, np.arange(1, 2*trial_num, 2), action_baseline+stage_2_selected_stimulus-1] = 1

            reward_masked = self.behav['reward'][b].copy()
            reward_masked[reward_masked==-1] = 0
            rew[b, np.arange(1, 2*trial_num, 2), 0] = reward_masked

            target[b, np.arange(0, 2*trial_num, 2)] = stage_1_selected_stimulus-1
            mask[b, np.arange(0, 2*trial_num, 2)] = self.behav['stage_1_selected_stimulus'][b] != -1
            target_mask[b, np.arange(0, 2*trial_num, 2), 0:2] = 0 # only first-stage action is valid
            target[b, np.arange(1, 2*trial_num, 2)] = action_baseline+stage_2_selected_stimulus-1
            mask[b, np.arange(1, 2*trial_num, 2)] = self.behav['stage_2_selected_stimulus'][b] != -1
            target_mask[b, 1::2][stage_2_state==2, 2:4] = 0 # only second-stage action in state 2 is valid
            target_mask[b, 1::2][stage_2_state==3, 4:6] = 0 # only second-stage action in state 3 is valid
        target_mask *= 9999
        device = 'cpu' if 'device' not in format_config else format_config['device']
        assert format_config['output_h0']
        include_embedding = 'include_embedding' in format_config and format_config['include_embedding']
        assert not include_embedding
        self.include_embedding = include_embedding
        act = torch.from_numpy(np.swapaxes(act, 0,1)).to(device=device)  # act shape: 2*trial_num, batch_size, act_num
        state = torch.from_numpy(np.swapaxes(state, 0,1)).to(device=device)  # state shape: 2*trial_num, batch_size, state_num
        rew = torch.from_numpy(np.swapaxes(rew, 0,1)).to(device=device)  # rew shape: 2*trial_num, batch_size, 1
        input = torch.cat([state, act, rew], -1) # input shape: 2*trial_num, batch_size, state_num+act_num+1=10

        target = torch.from_numpy(np.swapaxes(target, 0,1)).to(device=device)  # target shape: 2*trial_num, batch_size
        mask = torch.from_numpy(np.swapaxes(mask, 0,1)).to(device=device)  # mask shape: 2*trial_num, batch_size
        target_mask = torch.from_numpy(np.swapaxes(target_mask, 0,1)).to(device=device)  # target_mask shape: 2*trial_num, batch_size, act_num

        print(act.shape,rew.shape,input.shape,target.shape)
        self.torch_beahv_input = input.double()
        self.torch_beahv_target = target.long()
        self.torch_beahv_mask = (mask.double(), target_mask.double())

    def _augment_data(self, augment_level=1):
        """Augment data by flipping
        1) first-stage action and second-stage state
        2) second-stage actions
        3) both
        Only augment the data for the students. 4x"""
        behav = self.behav
        self.augment_ratio = 4
        number_block_before_aug = len(behav['action'])
        def _flip(data, value1, value2):
            data_new = data.copy()
            assert np.sum(data==value1)>0 and np.sum(data==value2)>0
            data_new[data==value1] = value2
            data_new[data==value2] = value1
            return data_new

        if augment_level >= 1: # most basic augmentation
            for b in range(number_block_before_aug): # flip first-stage action and second-stage state
                behav['stage_1_selected_stimulus'].append(_flip(self.behav['stage_1_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['stage_2_state'].append(_flip(self.behav['stage_2_state'][b], 2, 3)) # 2, 3, -1
                behav['stage_2_selected_stimulus'].append(self.behav['stage_2_selected_stimulus'][b]) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

            for b in range(number_block_before_aug): # flip both-second-stage actions
                behav['stage_1_selected_stimulus'].append(self.behav['stage_1_selected_stimulus'][b]) # 1, 2, -1
                behav['stage_2_state'].append(self.behav['stage_2_state'][b]) # 2, 3, -1
                behav['stage_2_selected_stimulus'].append(_flip(self.behav['stage_2_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

            for b in range(number_block_before_aug): # flip first-stage action and second-stage state, and both-second-stage actions
                behav['stage_1_selected_stimulus'].append(_flip(self.behav['stage_1_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['stage_2_state'].append(_flip(self.behav['stage_2_state'][b], 2, 3)) # 2, 3, -1
                behav['stage_2_selected_stimulus'].append(_flip(self.behav['stage_2_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

        if augment_level >= 2: # augmentation by partial flipping
            for b in range(number_block_before_aug): # flip second-stage-2 actions
                behav['stage_1_selected_stimulus'].append(self.behav['stage_1_selected_stimulus'][b]) # 1, 2, -1
                behav['stage_2_state'].append(self.behav['stage_2_state'][b]) # 2, 3, -1
                stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy() # 1, 2, -1
                stage_2_selected_stimulus_flipped = _flip(stage_2_selected_stimulus, 1, 2) # 1, 2, -1
                stage_2_selected_stimulus[self.behav['stage_2_state'][b] == 2] = stage_2_selected_stimulus_flipped[self.behav['stage_2_state'][b] == 2]
                behav['stage_2_selected_stimulus'].append(stage_2_selected_stimulus) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

            for b in range(number_block_before_aug):  # flip second-stage-3 actions
                behav['stage_1_selected_stimulus'].append(self.behav['stage_1_selected_stimulus'][b])  # 1, 2, -1
                behav['stage_2_state'].append(self.behav['stage_2_state'][b])  # 2, 3, -1
                stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy()  # 1, 2, -1
                stage_2_selected_stimulus_flipped = _flip(stage_2_selected_stimulus, 1, 2)  # 1, 2, -1
                stage_2_selected_stimulus[self.behav['stage_2_state'][b] == 3] = stage_2_selected_stimulus_flipped[self.behav['stage_2_state'][b] == 3]
                behav['stage_2_selected_stimulus'].append(stage_2_selected_stimulus)  # 1, 2, -1
                behav['reward'].append(behav['reward'][b])  # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

            for b in range(number_block_before_aug): # flip first-stage action and second-stage state, and second-stage-2 actions
                behav['stage_1_selected_stimulus'].append(_flip(self.behav['stage_1_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['stage_2_state'].append(_flip(self.behav['stage_2_state'][b], 2, 3)) # 2, 3, -1
                stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy() # 1, 2, -1
                stage_2_selected_stimulus_flipped = _flip(stage_2_selected_stimulus, 1, 2) # 1, 2, -1
                stage_2_selected_stimulus[self.behav['stage_2_state'][b] == 2] = stage_2_selected_stimulus_flipped[self.behav['stage_2_state'][b] == 2]
                behav['stage_2_selected_stimulus'].append(stage_2_selected_stimulus) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

            for b in range(number_block_before_aug): # flip first-stage action and second-stage state, and second-stage-3 actions
                behav['stage_1_selected_stimulus'].append(_flip(self.behav['stage_1_selected_stimulus'][b], 1, 2)) # 1, 2, -1
                behav['stage_2_state'].append(_flip(self.behav['stage_2_state'][b], 2, 3)) # 2, 3, -1
                stage_2_selected_stimulus = self.behav['stage_2_selected_stimulus'][b].copy() # 1, 2, -1
                stage_2_selected_stimulus_flipped = _flip(stage_2_selected_stimulus, 1, 2) # 1, 2, -1
                stage_2_selected_stimulus[self.behav['stage_2_state'][b] == 3] = stage_2_selected_stimulus_flipped[self.behav['stage_2_state'][b] == 3]
                behav['stage_2_selected_stimulus'].append(stage_2_selected_stimulus) # 1, 2, -1
                behav['reward'].append(behav['reward'][b]) # 0, 1, -1
                behav['sub_id'].append(behav['sub_id'][b])
                if 'task_id' in behav: behav['task_id'].append(behav['task_id'][b])
                behav['aug_block_number'].append(b)

        behav['action'] = behav['stage_1_selected_stimulus']

    def get_after_augmented_block_number(self, block_indices_before_augmentation):
        """Extract the block indices in augmented data, for each block-number-before-augmentation in block_indices_before_augmentation."""
        aug_block_number = np.array(self.behav['aug_block_number'])
        block_indices_after_augmentation = []
        for b in block_indices_before_augmentation: # for each block in the real data
            assert np.any(aug_block_number == b), (b, aug_block_number)
            aug_idx = np.where(aug_block_number == b)[0]
            block_indices_after_augmentation.append(aug_idx)
        block_indices_after_augmentation = list(np.concatenate(block_indices_after_augmentation))
        assert len(block_indices_after_augmentation) == len(block_indices_before_augmentation) * self.augment_ratio, (len(block_indices_after_augmentation), len(block_indices_before_augmentation), self.augment_ratio)
        return [int(x) for x in block_indices_after_augmentation] # convert to int, numpy.int64 is not json serializable



def _convert_array(array):
    try:
        # Try to convert to integer
        int_array = array.astype(int)
        # If conversion to int successful and no data loss, return int array
        if np.array_equal(int_array.astype(np.float64), array.astype(np.float64)):
            return int_array
    except ValueError:
        pass
    try:
        # Try to convert to float
        return array.astype(np.float64)
    except ValueError:
        pass
    # array of str to lower case
    return np.array([item.lower() for item in array])


def load_subject_data_csv(file_path):
    header = [
        "trial_num",
        "drift_1",
        "drift_2",
        "drift_3",
        "drift_4",
        "stage_1_response",
        "stage_1_selected_stimulus",
        "stage_1_RT",
        "transition",
        "stage_2_response",
        "stage_2_selected_stimulus",
        "stage_2_state",
        "stage_2_RT",
        "reward",
        "redundant_task_variable",
    ]

    task_dict = {col: [] for col in header}

    with open(file_path, "r") as f:
        # Initialize a flag to indicate whether we've started the task data
        start_task_data = False

        for line in f:
            row = line.strip().split(",")

            # If the first column of the row equals '1', start appending data
            if row[0] == '1':
                start_task_data = True

            if start_task_data:
                for col, value in zip(header, row):
                    task_dict[col].append(value)

    # Convert lists in the dict to numpy arrays
    for key in task_dict:
        task_dict[key] = _convert_array(np.array(task_dict[key]))

    assert np.all(task_dict['trial_num'] == np.arange(1, 201)), task_dict['trial_num']
    for i in range(1, 5):
        assert np.all(0 <= task_dict[f'drift_{i}']), task_dict[f'drift_{i}']
        assert np.all(task_dict[f'drift_{i}'] <= 1), task_dict[f'drift_{i}']
    assert np.all(task_dict['redundant_task_variable'] == 1), task_dict['redundant_task_variable']
    # assert np.all(task_dict['stage_1_RT'] >= -1), task_dict['stage_1_RT'] # some RT is problematic, not use it
    # assert np.all(task_dict['stage_2_RT'] >= -1), task_dict['stage_2_RT']

    assert np.all(np.isin(task_dict['stage_1_response'], ['left', 'right', '-1'])), task_dict['stage_1_response']
    assert np.all(np.isin(task_dict['stage_1_selected_stimulus'], [1, 2, -1])), task_dict['stage_1_selected_stimulus']
    stage_1_selected_stimulus_mask = ((task_dict['stage_1_response'] != '-1') & (task_dict['stage_1_selected_stimulus'] != -1))
    task_dict['stage_1_selected_stimulus'][~stage_1_selected_stimulus_mask] = -1

    # make sure stage_1_response and stage_1_selected_stimulus are consistent
    if not np.all(task_dict['stage_1_response'][stage_1_selected_stimulus_mask] == np.where(task_dict['stage_1_selected_stimulus'][stage_1_selected_stimulus_mask] == 1, 'left', 'right')):
        idx = np.argwhere(task_dict['stage_1_response'] != np.where(task_dict['stage_1_selected_stimulus'] == 1, 'left', 'right'))
        print(len(idx.flatten()))

    assert np.all(np.isin(task_dict['transition'], ['true', 'false', '-1'])), task_dict['transition']
    for i in range(len(task_dict['transition'])): # make sure transition is consistent with stage_1_selected_stimulus and stage_2_state
        if task_dict['transition'][i] == 'true':
            assert task_dict['stage_1_selected_stimulus'][i] == task_dict['stage_2_state'][i]-1, (task_dict['stage_1_selected_stimulus'][i], task_dict['stage_2_state'][i])
        elif task_dict['transition'][i] == 'false':
            assert task_dict['stage_1_selected_stimulus'][i] != task_dict['stage_2_state'][i]-1, (task_dict['stage_1_selected_stimulus'][i], task_dict['stage_2_state'][i])
        else:
            pass
    transition_mask = (task_dict['transition'] != '-1')

    assert np.all(np.isin(task_dict['stage_2_response'], ['left', 'right','-1'])), task_dict['stage_2_response']
    assert np.all(np.isin(task_dict['stage_2_selected_stimulus'], [1,2, -1])), task_dict['stage_2_selected_stimulus']
    stage_2_selected_stimulus_mask = ((task_dict['stage_2_response'] != '-1') & (task_dict['stage_2_selected_stimulus'] != -1))
    task_dict['stage_2_selected_stimulus'][~stage_2_selected_stimulus_mask] = -1

    # make sure stage_2_response and stage_2_selected_stimulus are consistent
    if not np.all(task_dict['stage_2_response'][stage_2_selected_stimulus_mask] == np.where(task_dict['stage_2_selected_stimulus'][stage_2_selected_stimulus_mask] == 1, 'left', 'right')):
        idx = np.argwhere(task_dict['stage_2_response'] != np.where(task_dict['stage_2_selected_stimulus'] == 1, 'left', 'right'))
        print(len(idx.flatten()))

    assert np.all(np.isin(task_dict['stage_2_state'], [2,3, -1])), task_dict['stage_2_state']
    stage_2_state_mask = (task_dict['stage_2_state'] != -1)

    assert np.all(np.isin(task_dict['reward'], [0,1, -1])), task_dict['reward']
    reward_mask = (task_dict['reward'] != -1)

    # if earlier events are masked, later events should be masked too; then the reward_mask is the strictest
    assert np.all(stage_1_selected_stimulus_mask[transition_mask]), (np.argwhere(~stage_1_selected_stimulus_mask), np.argwhere(~transition_mask))
    assert np.all(transition_mask == stage_2_state_mask), (np.argwhere(~transition_mask), np.argwhere(~stage_2_state_mask))
    assert np.all(stage_2_state_mask[stage_2_selected_stimulus_mask]), (np.argwhere(~stage_2_state_mask), np.argwhere(~stage_2_selected_stimulus_mask))
    assert np.all(stage_2_selected_stimulus_mask[reward_mask]), (np.argwhere(~stage_2_selected_stimulus_mask), np.argwhere(~reward_mask))
    bad_trials = [np.sum(~stage_1_selected_stimulus_mask), np.sum(~transition_mask), np.sum(~stage_2_state_mask), np.sum(~stage_2_selected_stimulus_mask), np.sum(~reward_mask)]
    assert np.sum(~stage_1_selected_stimulus_mask) <= np.sum(~transition_mask) <= np.sum(~stage_2_state_mask) <= np.sum(~stage_2_selected_stimulus_mask) <= np.sum(~reward_mask), bad_trials
    # if np.max(bad_trials):
    #     print('bad trials', bad_trials)
    masks = {
        'stage_1_selected_stimulus_mask': stage_1_selected_stimulus_mask,
        'transition_mask': transition_mask,
        'stage_2_state_mask': stage_2_state_mask,
        'stage_2_selected_stimulus_mask': stage_2_selected_stimulus_mask,
        'reward_mask': reward_mask,
    }
    for key in [
        "trial_num",
        "stage_1_response",
        # "stage_1_RT",
        "stage_2_response",
        # "stage_2_RT",
        "redundant_task_variable",
    ]:
        task_dict.pop(key)
    return task_dict | masks


def _combine_subjects_data(data_path):
    import pandas as pd

    file_path = (DATA_PATH / 'Gillandata/Experiment 1/self_report_study1.csv')
    df1 = pd.read_csv(file_path)
    df1 = df1.rename(columns={'Unnamed: 0': 'sub_id', 'subj.x': 'subj'})
    df1['filepath'] = 'Experiment 1/twostep_data_study1'

    file_path = (DATA_PATH / 'Gillandata/Experiment 2/self_report_study2.csv')
    df2 = pd.read_csv(file_path)
    df2 = df2.rename(columns={'Unnamed: 0': 'sub_id'})
    df2['sub_id'] += df1['sub_id'].max()
    df2['filepath'] = 'Experiment 2/twostep_data_study2'
    df = pd.concat([df1, df2], axis=0)
    df['sub_id'] -= 1
    behav = {}
    subject_remapping = {}
    for i, row in df.iterrows():
        sub_id = row['sub_id']
        subj = row['subj']
        subject_remapping[subj] = sub_id
        print(sub_id, subj)
        task_dict = load_subject_data_csv(os.path.join(DATA_PATH / 'Gillandata',
                                              row['filepath'],
                                              subj+'.csv'
                                              ))
        behav.setdefault('sub_id', []).append(sub_id)
        for key, value in task_dict.items():
            behav.setdefault(key, []).append(value)
    joblib.dump([subject_remapping, behav], data_path)