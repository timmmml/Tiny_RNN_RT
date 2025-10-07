import numpy as np
from .BaseTwoStepDataset import BaseTwoStepDataset, _combine_data_dict, _segment_long_block
import sys
import joblib

class AkamRatDataset(BaseTwoStepDataset):
    """A dataset class for the advanced two-step task with binary actions, states, and rewards.

    Akam's rats.
    Load preprocessed data generated from his Python code. See analysis_code/two_step/Two_step.py.

    Attributes:
         unique_trial_type: How many possible unique trial observations (actions * states * rewards = 8 combinations)
         behav: Standard format of behavioral data.
         data_path: Where to load the data.
         behav_format: tensor (for RNN) or cog_session (for Cog agents)?
         torch_beahv_input: tensor format of agent's input
         torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
         torch_beahv_target: tensor format of agent's target output
         cog_sessions: cog_session format of agent's input & target output
         batch_size: How many blocks are there in the current loaded data?
    """
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=False):
        """Initialize the dataset."""
        if 'task' not in behav_data_spec:
            self.task = behav_data_spec['task'] = 'baseline'
        else:
            self.task = behav_data_spec['task']
        if self.task == 'reversal_learning': # [358 359 360 361 367 368 380 382 383 388]
            self._data_dict_list = joblib.load(data_path / 'two_step_ACC_reversal_learning.pkl')
            self.unique_trial_type = 4
        elif self.task == 'no_transition_reversal': # [269 270 271 272 273 274 275 277 278 279]
            self._data_dict_list = joblib.load(data_path / 'two_step_ACC_no_transition_reversal.pkl')
            self.unique_trial_type = 8
        elif self.task == 'baseline':
            self._data_dict_list = joblib.load(data_path / 'two_step_ACC_baseline_sessions.pkl')
            self.unique_trial_type = 8
        else:
            raise ValueError('Invalid task name.')
        self.subject_IDs = np.unique([s['subject_ID'] for s in self._data_dict_list])
        self.subject_remapping = {self.subject_IDs[i]: i for i in range(len(self.subject_IDs))}
        if verbose:
            print('subject_IDs', self.subject_IDs)
        super().__init__(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)

    def data_summary(self):
        """Print a summary of the dataset.
        animal_name: total trial num,
        49: 18426, 50: 20479, 51: 21006, 52: 14815, 53: 16691, 54: 14328, 100: 9485,
        95: 8865, 96: 7088, 97: 9121, 98: 9711, 99: 7775,
        263: 15233, 264: 13201, 266: 14671, 267: 18285, 268: 11057

        Key in each session:
            'subject_ID':
            'date':
            'fraction_rewarded':
            'nTrials':
            'choices':
            'outcomes':
            'second_steps':
            'transitions':
            'trial_trans_rule':  transition state/rule for each trial
                =1: A0->S0, A1->S1 common transition
                =0: A0->S1, A1->S0 common transition
            'trial_rew_rule' :  reward state for each trial
                =0: S0->R0, S1->R1 larger probability
                =1: neutral
                =2: S0->R1, S1->R0 larger probability
            'block_start_trials': start trial for each block
            'block_end_trials': end trial for each block
            'block_reward_rule': reward state for each block
            'block_transition_rule': transition state for each block

        """
        import pprint
        pp = pprint.PrettyPrinter(depth=4, sort_dicts=False)
        trial_counter = {}
        subject_summary = {}
        for s in self._data_dict_list:
            print(s['subject_ID'], s['date'], 'fraction_rewarded', s['fraction_rewarded'],'nTrials', s['nTrials'])
            if s['subject_ID'] not in subject_summary:
                subject_summary[s['subject_ID']] = {
                    'subject_ID': s['subject_ID'],
                    'date': s['date'],
                    'nTrials': 0,
                    'total_reward': 0,
                    'non_neutral_block_length': [],
                }
            block_length = np.array(s['block_end_trials']) - np.array(s['block_start_trials'])
            block_length = block_length[s['block_reward_rule'] != 1]
            subject_summary[s['subject_ID']]['non_neutral_block_length'].append(block_length)
            subject_summary[s['subject_ID']]['nTrials'] += s['nTrials']
            subject_summary[s['subject_ID']]['total_reward'] += s['fraction_rewarded'] * s['nTrials']
            trial_counter.setdefault(s['subject_ID'], []).append(s['nTrials'])

        pp.pprint({k: np.sum(v) for k, v in trial_counter.items()})
        import pandas as pd
        subject_summary = pd.DataFrame(list(subject_summary.values()))
        subject_summary['non_neutral_block_length'] = subject_summary['non_neutral_block_length'].apply(lambda x: np.concatenate(x))
        subject_summary['non_neutral_block_num'] = subject_summary['non_neutral_block_length'].apply(lambda x: len(x))
        subject_summary['non_neutral_block_length'] = subject_summary['non_neutral_block_length'].apply(lambda x: np.mean(x))
        subject_summary['average_reward'] = subject_summary['total_reward'] / subject_summary['nTrials']
        return subject_summary

    def _detect_trial_type(self):
        """Detect trial type from behavioral data."""
        behav = self.behav
        behav['trial_type'] = []
        for i in range(len(behav['action'])):
            if self.task == 'reversal_learning':
                behav['trial_type'].append(behav['stage2'][i] * 2 + behav['reward'][i])
            else:
                behav['trial_type'].append(behav['action'][i] * 4 + behav['stage2'][i] * 2 + behav['reward'][i])

    def _load_all_trial_type(self, behav_data_spec):
        np2list = lambda L: [np.array([x]) for x in L]
        if self.task == 'reversal_learning':
            self.behav = {
                'action': np2list([0, 0, 1, 1]),
                'stage2': np2list([0, 0, 1, 1]),
                'reward': np2list([0, 1, 0, 1]),
            }
        else:
            self.behav = {
                'action': np2list([0,0,0,0,1,1,1,1]),
                'stage2': np2list([0,0,1,1,0,0,1,1]),
                'reward': np2list([0,1,0,1,0,1,0,1]),
            }
        self._detect_trial_type()

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load data from disk following data specifications."""
        if 'all_trial_type' in behav_data_spec and behav_data_spec['all_trial_type']:
            self._load_all_trial_type(behav_data_spec)
            return

        loaded_sessions = []
        if 'animal_name' in behav_data_spec and 'session_name' not in behav_data_spec:
            for s in self._data_dict_list:
                if behav_data_spec['animal_name'] == 'all' or s['subject_ID'] == int(behav_data_spec['animal_name']):
                    loaded_sessions.append(s)
        else:
            raise ValueError

        if neuro_data_spec is None:
            neuro_data_spec = {}
        self.behav_data_spec = behav_data_spec
        self.neuro_data_spec = neuro_data_spec

        self.behav = behav = {}
        self.neuro = neuro = {}

        for s in loaded_sessions:
            if verbose: print(s['subject_ID'], s['date'], 'fraction_rewarded', s['fraction_rewarded'],'nTrials', s['nTrials'], end=',')
            sub_id = self.subject_remapping[s['subject_ID']]
            nTrials = s['nTrials']
            if 'block_truncation' in behav_data_spec:
                bt = behav_data_spec['block_truncation']
                if nTrials < bt[1]:
                    if verbose: print(f'WARNING: session {s.date} have {nTrials} trials fewer than {bt[1]}; IGNORED')
                    continue
                eff_trials = np.arange(bt[0], bt[1])
            else:
                eff_trials = np.arange(0, nTrials)
            if 'max_segment_length' in behav_data_spec:
                max_segment_length = behav_data_spec['max_segment_length']
            else:
                max_segment_length = None

            action_segments = _segment_long_block(s['choices'][eff_trials], max_trial_num=max_segment_length, verbose=verbose)
            if self.task == 'reversal_learning':
                stage2_segments = action_segments
            else:
                stage2_segments = _segment_long_block(s['second_steps'][eff_trials], max_trial_num=max_segment_length)
                transition_segments = _segment_long_block(s['transitions'][eff_trials], max_trial_num=max_segment_length)
                behav.setdefault('transitions', []).extend(transition_segments)
            reward_segments = _segment_long_block(s['outcomes'][eff_trials], max_trial_num=max_segment_length)
            if verbose: print('')
            behav.setdefault('action', []).extend(action_segments) # list of 1d array
            behav.setdefault('stage2', []).extend(stage2_segments)
            behav.setdefault('reward', []).extend(reward_segments)
            behav.setdefault('sub_id', []).extend([sub_id] * len(action_segments))
        behav['aug_block_number'] = list(np.arange(len(behav['action']))) # list of numbers, the block number before augmentation
        if 'augment' in behav_data_spec and behav_data_spec['augment']:
            self._augment_data()
        self._detect_trial_type()
        print("===loaded all===")
        print('Total batch size:', self.batch_size)
        print('Total trial num:', self.total_trial_num)

    def _augment_data(self):
        """Augment data by flipping action and stage2."""
        behav = self.behav
        number_block_before_aug = len(behav['action'])
        for b in range(number_block_before_aug): # flip action
            behav['action'].append(1 - behav['action'][b])
            behav['stage2'].append(behav['stage2'][b])
            behav['reward'].append(behav['reward'][b])
            behav['sub_id'].append(behav['sub_id'][b])
            behav['aug_block_number'].append(b)
        for b in range(number_block_before_aug): # flip stage2
            behav['action'].append(behav['action'][b])
            behav['stage2'].append(1 - behav['stage2'][b])
            behav['reward'].append(behav['reward'][b])
            behav['sub_id'].append(behav['sub_id'][b])
            behav['aug_block_number'].append(b)
        for b in range(number_block_before_aug): # flip both
            behav['action'].append(1 - behav['action'][b])
            behav['stage2'].append(1 - behav['stage2'][b])
            behav['reward'].append(behav['reward'][b])
            behav['sub_id'].append(behav['sub_id'][b])
            behav['aug_block_number'].append(b)

    def get_after_augmented_block_number(self, block_indices_before_augmentation):
        """Extract the block indices after augmentation with the block-number-before-augmentation in block_indices_before_augmentation."""
        aug_block_number = np.array(self.behav['aug_block_number'])
        block_indices_after_augmentation = []
        for b in block_indices_before_augmentation: # for each block in the real data
            aug_idx = np.where(aug_block_number == b)[0]
            block_indices_after_augmentation.append(aug_idx)
        block_indices_after_augmentation = list(np.concatenate(block_indices_after_augmentation))
        assert len(block_indices_after_augmentation) == len(block_indices_before_augmentation) * 4
        return  [int(x) for x in block_indices_after_augmentation] # convert to int, numpy.int64 is not json serializable

