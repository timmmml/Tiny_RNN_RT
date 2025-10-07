import numpy as np
import torch
import torch.nn as nn

class BaseTwoStepDataset(object):
    """A dataset class suitable for several two-step tasks with binary actions, states, rewards;
    also for one step task with binary actions and rewards.

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
        """Initialize the dataset."""
        if not hasattr(self, 'unique_trial_type'):
            self.unique_trial_type = None
        assert data_path is not None
        self.behav = {}
        self.torch_beahv_input = None
        self.torch_beahv_input_1hot = None
        self.torch_beahv_target = None
        self.torch_beahv_mask = None
        self.cog_sessions = None
        self.behav_format = None
        self.data_path = data_path
        if behav_data_spec is not None:
            self.load_data(behav_data_spec, neuro_data_spec, verbose)
        return

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        """Load data from disk following data specifications.

        The loaded data is stored in self.behav, the standard format of behavioral data.
            {'action':List[1d array], 'stage2':List[1d array], 'reward':List[1d array]}
        Neural info stored in self.neuro. The format depends on the dataset.

        Args:
            behav_data_spec: A dict specifies which behavioral data to load in which format.
                e.g., which animal, recording session, or block to load from. The format is dataset dependent.
            neuro_data_spec: A dict specifies which neural data to load in which format.
        """
        raise NotImplementedError

    def behav_to(self, format_config=None):
        """Transform standard data to specified data format, either tensor or cog_session format.

        Args:
            format_config: A dict specifies how the standard data should be transformed.
                Only the "behav_format" and "one_hot" keys are used.

        Returns:
            The dataset instance itself.
        """
        assert format_config is not None
        self.behav_format = format_config['behav_format']
        if self.behav_format == 'tensor':
            self._behav_to_tensor(format_config)
        elif self.behav_format == 'cog_session':
            self._behav_to_cog_sessions(format_config)
        else:
            raise NotImplementedError
        return self

    @property
    def batch_size(self):
        """How many blocks in total?"""
        return len(self.behav['action'])

    @property
    def total_trial_num(self):
        """How many trials in total?"""
        return sum([len(block) for block in self.behav['action']])


    def get_behav_data(self, batch_indices, format_config=None):
        """Return a subset (a few batches) of loaded data with specified format.

        Args:
            batch_indices: A list or 1d ndarray specifies which batches are extracted.
            format_config: A dict specifies how the standard data should be transformed.
                Mainly "behav_format" and "one_hot" keys.

        Returns:
            A dict contains:
                input, target_output, mask if for the tensor format.
                List of sessions (a session = a block) if for the cog_session format.
        """
        assert isinstance(batch_indices, list) or isinstance(batch_indices, np.ndarray) and len(batch_indices.shape) == 1
        if self.behav_format is None: # the standard format
            behav = {}
            for k, v in self.behav.items():
                behav[k] = [v[idx] for idx in batch_indices]
            return behav
        assert format_config is not None
        assert self.behav_format == format_config['behav_format']
        trial_type = np.array(self.behav['trial_type'])
        if self.behav_format == 'tensor':
            mask = self.torch_beahv_mask[:,batch_indices]
            mask_block_sum = mask.sum(dim=1) # shape (trial_num,)
            last_1_idx = torch.where(mask_block_sum>0)[0][-1].item()
            trial_slice = slice(0, last_1_idx+1) # remove the last few trials with mask=0 for the whole batch
            if 'one_hot' in format_config and format_config['one_hot']:
                return {'input': self.torch_beahv_input_1hot[trial_slice,batch_indices],
                        'target': self.torch_beahv_target[trial_slice,batch_indices],
                        'mask': self.torch_beahv_mask[trial_slice,batch_indices],
                        'trial_type': trial_type[batch_indices]
                        }
            else:
                return {'input': self.torch_beahv_input[trial_slice,batch_indices],
                        'target': self.torch_beahv_target[trial_slice,batch_indices],
                        'mask': self.torch_beahv_mask[trial_slice, batch_indices],
                        'trial_type': trial_type[batch_indices]
                        }
        elif self.behav_format == 'cog_session':
            return {'input': [self.cog_sessions[b] for b in batch_indices],
                    'trial_type': trial_type[batch_indices]
                    }
        else:
            raise NotImplementedError

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
        max_trial_num = max([len(block) for block in self.behav['reward']])
        include_embedding = 'include_embedding' in format_config and format_config['include_embedding']
        self.include_embedding = include_embedding

        act = np.zeros((self.batch_size, max_trial_num))
        rew = np.zeros((self.batch_size, max_trial_num))
        stage2 = np.zeros((self.batch_size, max_trial_num))
        sub = np.zeros((self.batch_size, max_trial_num))
        mask = np.zeros((self.batch_size, max_trial_num))
        for b in range(self.batch_size):
            this_trial_num = len(self.behav['reward'][b])
            mask[b, :this_trial_num] = 1
            act[b, :this_trial_num] = self.behav['action'][b]
            rew[b, :this_trial_num] = self.behav['reward'][b]
            stage2[b, :this_trial_num] = self.behav['stage2'][b]
            if include_embedding:
                this_sub_id = np.unique(self.behav['sub_id'][b])
                assert len(this_sub_id) == 1
                sub[b, :] = this_sub_id
        device = 'cpu' if 'device' not in format_config else format_config['device']
        output_h0 = True if 'output_h0' not in format_config else format_config['output_h0']

        act = torch.from_numpy(act.T[..., None]).to(device=device)  # act shape: trial_num, batch_size, 1
        stage2 = torch.from_numpy(stage2.T[..., None]).to(device=device)
        rew = torch.from_numpy(rew.T[..., None]).to(device=device)
        if output_h0:
            input = torch.cat([act, stage2, rew], -1)  # trial_num, batch_size, input_size=3
            target = act[:, :, 0]  # class, not one-hot
            # print('output_h0', output_h0, 'h0 included in target')
        else:
            input = torch.cat([act, stage2, rew], -1)[:-1]
            target = act[1:, :, 0]  # class, not one-hot
            # print('output_h0', output_h0, 'h0 excluded in target')
        # print(act.shape,rew.shape,input.shape,target.shape)

        if include_embedding:
            sub = torch.from_numpy(np.swapaxes(sub[..., None], 0,1)).to(device=device)
            input = torch.cat([input, sub], -1) # input shape: trial_num, batch_size, 3+sub_num

        self.torch_beahv_input = input.double()
        self.torch_beahv_target = target.long()
        self.torch_beahv_mask = torch.from_numpy(mask.T).to(device=device).double()
        if self.unique_trial_type == 4: # only stage2 & reward
            trial_type = input[..., 1] * 2 + input[..., 2]
        elif self.unique_trial_type == 8: # action & stage2 & reward
            trial_type = input[..., 0] * 4 + input[..., 1] * 2 + input[..., 2]
        elif self.unique_trial_type == -1:
            return # no one-hot encoding
        else:
            raise NotImplementedError
        self.torch_beahv_input_1hot = nn.functional.one_hot(trial_type.to(torch.int64), num_classes=self.unique_trial_type).double()


    def _behav_to_cog_sessions(self, format_config):
        """Transform standard behavioral format to cog_session format, stored in cog_sessions attribute.

        Args:
            format_config: A dict specifies how the standard data should be transformed.
        """
        if self.cog_sessions is not None:
            return
        behav = self.behav
        action = behav['action']
        reward = behav['reward']
        stage2 = behav['stage2'] if 'stage2' in behav else action
        transitions = None if 'transitions' not in behav else behav['transitions']
        episode_num = len(reward)

        self.cog_sessions = []
        print('Transforming standard format to cog_session format...')
        print('(episode, trial_num):', end='')
        for epi_idx in range(episode_num):
            trial_num = len(reward[epi_idx])
            print(f'({epi_idx}, {trial_num}),', end='')
            self.cog_sessions.append(
                nn_session(action[epi_idx], stage2[epi_idx], reward[epi_idx], trial_num, transitions=transitions))
        print('\nTotal block num', episode_num)

def _combine_data_dict(old, new):
    """Append the new dict's value to the old dict's list of values.
    Used when combining data from many recording files.
    In-place update of old dict.

    Args:
        new: A dict, each key has a value.
        old: A dict, each key has a list of values.

    Return:
    """
    for k,v in new.items():
        if k not in old:
            old[k] = []
        old[k].append(v)


def _segment_long_block(block, max_trial_num=150, verbose=False):
    """Segment a long block into several blocks with length smaller than max_trial_num.
    The can accelerate the training process of RNN models
    .
    Args:
        block: A 1d numpy array.
        max_trial_num: An int, the maximum length of a block.

    Return:
        A list of 1d numpy array
    """
    trial_num = len(block)
    if max_trial_num is None or trial_num <= max_trial_num:
        return [block]
    else:
        block_list = []
        segment_num = int(np.ceil(trial_num / max_trial_num))
        segment_length = int(np.ceil(trial_num / segment_num))
        if verbose: print(f'segmenting into',end=' ')
        for i in range(0, segment_num):
            block_list.append(block[i * segment_length: (i + 1) * segment_length])
            if verbose: print(len(block_list[-1]),end=' ')
        assert segment_num * segment_length >= trial_num
        assert sum([len(b) for b in block_list]) == trial_num
        return block_list


class nn_session():
    """Session class for Cog agents.
    A session is a block of trials.

    Attributes:
        n_trials: Number of trials in this session.
        CTSO: A dict; each key is one type of features across all trials
    """
    def __init__(self, choices, second_steps, outcomes, n_trials=100, choice_pr=None,transitions=None):
        self.n_trials = n_trials
        choices_s = None
        self.CTSO = {'choices': choices,
                     'transitions': transitions if transitions is not None else (choices == second_steps).astype(int),
                     'second_steps': second_steps,
                     'choices_s': choices_s,
                     'outcomes': outcomes,
                     'choice_pr': choice_pr}

    def unpack_trial_data(self, order='CTSO', dtype=int):
        """Return elements of trial_data dictionary in specified order and data type.
        """
        o_dict = {'C': 'choices', 'T': 'transitions', 'S': 'second_steps', 'O': 'outcomes'}
        if dtype == int:
            return [self.CTSO[o_dict[i]] for i in order]
        else:
            return [self.CTSO[o_dict[i]].astype(dtype) for i in order]