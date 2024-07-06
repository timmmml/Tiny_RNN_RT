from .BaseAgent import BaseAgent
from .network_models import RNNnet, set_seed
from copy import deepcopy
import torch
import os
import json
import joblib
from path_settings import *
import numpy as np
import torch.nn as nn


class RNNAgent(BaseAgent):
    """All types of RNN agents.

    Attributes:
        config: everything
        model: the RNN network, implemented in PyTorch way.
        behav_loss_function: a loss function for behavior prediction
        num_params: total number of parameters in the model

    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        assert config['rnn_type'] in ['GRU', 'SGRU', 'MIGRU', 'PNR1', 'PNR2','LR']
        other_config = deepcopy(config)
        [other_config.pop(key) for key in ['input_dim', 'hidden_dim', 'output_dim']]
        set_seed(config['seed'])
        self.rnn_type = config['rnn_type']
        self.model = RNNnet(config['input_dim'], config['hidden_dim'], config['output_dim'], **other_config).double()
        self.model.share_memory() # required for multiprocessing
        self.num_params = sum(param.numel() for param in self.model.parameters())
        self.behav_loss_function = nn.CrossEntropyLoss(reduction='none')
        if 'device' in config:
            self.model.to(config['device'])

    def load(self, model_path=None, strict=True):
        """Load model parameters from disk."""
        if model_path is None:
            model_path = self.config['model_path']
        state_dict = torch.load(MODEL_SAVE_PATH / model_path / 'model.ckpt', map_location=torch.device(self.config['device']))
        self.model.load_state_dict(state_dict, strict=strict)
        self.model.eval()

    def forward(self, input, h0=None, standard_output=False):
        """Process a batch of input.

        Args:
            input: shape: seq_len, batch_size, input_dim; or a dict containing 'input' and 'mask'
            h0: initial hidden state
            standard_output: if True, returns a dict of list
                each list includes the output of each episode
        Returns:
            a dict of internal states and outputs
        """
        if isinstance(input, dict):
            input_ = input['input']
        else:
            input_ = input
        if h0 is not None:
            assert len(h0.shape) == 1
            h0 = h0[None, None, :]
            if isinstance(h0, np.ndarray):
                h0 = torch.from_numpy(h0).to(device=input_.device)
        scores, rnn_out = self.model(input_, get_rnnout=True, h0=h0)
        if standard_output:
            assert 'mask' in input
            mask = input['mask'].detach().cpu().numpy() # shape: seq_len, batch_size
            scores_list = []
            rnn_out_list = []
            scores = scores.detach().cpu().numpy()
            rnn_out = rnn_out.detach().cpu().numpy()
            for i in range(mask.shape[1]):
                # make sure 0 in mask only appears at the end of each episode
                if np.any(mask[:, i] == 0):
                    first_zero_loc = np.where(mask[:, i] == 0)[0][0]
                    if not np.all(mask[first_zero_loc:, i] == 0):
                        print('Warning: there are non-zero values after 0 in mask, be careful when analyzing the results.')
                    last_one_loc = np.where(mask[:, i] == 1)[0][-1]
                    # make mask 1 before the last_one_loc, and 0 after the last_one_loc
                    mask[:last_one_loc, i] = 1
                scores_list.append(scores[:int(mask[:, i].sum()) + 1, i, :])
                rnn_out_list.append(rnn_out[:int(mask[:, i].sum()) + 1, i, :])
            scores = scores_list
            rnn_out = rnn_out_list
        return {'output': scores, 'internal': rnn_out}

    def save(self, params=None, verbose=False):
        """Save config, model, and results."""
        model_path = self.config['model_path']
        os.makedirs(MODEL_SAVE_PATH / model_path, exist_ok=True)
        # save config
        self.save_config()

        # save model parameters
        if params is None: # current state dict is saved
            torch.save(self.model.state_dict(), MODEL_SAVE_PATH / model_path / 'model.ckpt')
        else:
            torch.save(params, MODEL_SAVE_PATH / model_path / 'model.ckpt')


        if verbose: print('Saved model at', MODEL_SAVE_PATH / model_path)

    def _eval_1step(self, input, target, mask, h0=None):
        """Return loss on test/val data set without gradient update."""
        with torch.no_grad():
            model_pass = self._compare_to_target(input, target, mask, h0=h0)
            # self._to_numpy(model_pass)
        return model_pass

    def _compare_to_target(self, input, target, mask, h0=None):
        """Compare model's output to target and compute losses.

        Args:
            input: shape (seq_len, batch_size, input_dim), 0 pading for shorter sequences
            target: shape (seq_len, batch_size) or (seq_len, batch_size, output_dim), 0 pading for shorter sequences
            mask: shape (seq_len, batch_size), 1 for valid, 0 for invalid
            h0: initial hidden state of the model.
        Returns:
            pass: a dict containing all the information of this pass.
        """
        model_pass = self.forward(input, h0=h0)
        scores = model_pass['output'] # shape: seq_len, batch_size, output_dim
        rnn_out = model_pass['internal']
        if not isinstance(target, dict):
            multi_target = {1: target} # fit to multiple targets, key=loss proportion, value=target
        else:
            multi_target = target

        assert np.sum(list(multi_target.keys())) == 1, 'loss proportion should sum to 1'
        target = multi_target[list(multi_target.keys())[0]]
        if self.config['output_h0']:
            assert scores.shape[0] - 1 == target.shape[0], (scores.shape, target.shape)
            # because output_h0 is True, scores will have one more time dimension than target/input/mask
            scores = scores[:-1] # remove the last time dimension
            rnn_out = rnn_out[:-1]
        else:
            assert scores.shape[0]  == target.shape[0]
        # WARNING: maybe mask out score and rnn_out here is not necessary
        scores = scores * mask[..., None] # mask out one additional time step
        rnn_out = rnn_out * mask[..., None]
        model_pass['output'] = scores
        model_pass['internal'] = rnn_out
        # scores.shape: trial_num, batch_size, output_dim
        # target.shape: trial_num, batch_size
        output_dim = self.config['output_dim']
        assert scores.shape[-1] == output_dim
        # print('target', target)
        for loss_proportion, target in multi_target.items():
            if len(target.shape) == 2: # (seq_len, batch_size)
                target_temp = target.flatten()
                loss_shape = target.shape
            elif len(target.shape) == 3: # (seq_len, batch_size, output_dim)
                # assume already probability here
                target_temp = target.reshape([-1, output_dim])
                loss_shape = target.shape[:-1]
            else:
                raise ValueError('target should be 2 or 3 dimensional.')
            behav_loss_total = self.behav_loss_function(scores.reshape([-1, output_dim]), target_temp).reshape(loss_shape)
            behav_loss_total = behav_loss_total * mask
            total_trial_num = torch.sum(mask)
            behav_loss = behav_loss_total.sum() / total_trial_num # only average over valid trials
            if 'behav_loss_total' not in model_pass:
                model_pass['behav_loss_total'] = behav_loss_total * loss_proportion
                model_pass['behav_loss'] = behav_loss * loss_proportion
            else:
                model_pass['behav_loss_total'] += behav_loss_total * loss_proportion
                model_pass['behav_loss'] += behav_loss * loss_proportion

        model_pass['behav_mask_total'] = mask
        model_pass['total_trial_num'] = total_trial_num
        return model_pass

    def simulate(self, task, config, save=True):
        """Simulate the agent's behavior on the task.

        Parameters should be set before this function; otherwise, the default parameters will be used.
        TODO: not implemented yet
        Args:
            task: the task instance.
            config: the config dict.
                n_blocks: number of blocks to simulate.
                n_trials: number of trials in each block.
                sim_seed: random seed for simulation.
                sim_exp_name: the name of the experiment when saving the results.
                additional_name: additional name to add to the file name when saving the results.
            save: whether to save the results to disk.

        Returns:
            A dictionary of simulation results.
        """
        print('Simulating RNN agent', self.rnn_type)
        n_blocks = config['n_blocks']
        n_trials = config['n_trials']
        sim_seed = config['sim_seed']
        sim_exp_name = config['sim_exp_name']
        additional_name = config['additional_name']
        if len(sim_exp_name) == 0 and save:
            raise ValueError('sim_exp_name must be specified if save is True')
        print('n_blocks', n_blocks, 'n_trials', n_trials, 'sim_seed', sim_seed, 'sim_exp_name', sim_exp_name, 'additional_name', additional_name)

        behav = {}
        behav['action'] = []
        behav['stage2'] = []
        behav['reward'] = []
        behav['mid_vars'] = []

        set_seed(sim_seed)

        for _ in range(n_blocks):
            DVs = self.model.simulate(task, n_trials, get_DVs=True)
            behav['action'].append(DVs['choices'])
            behav['stage2'].append(DVs['second_steps'])
            behav['reward'].append(DVs['outcomes'])
            for k in ['choices', 'second_steps', 'outcomes']:
                DVs.pop(k)
            behav['mid_vars'].append(DVs)
        if save:
            if len(additional_name) and additional_name[0] != '_':
                additional_name = '_' + additional_name
            sim_path = SIM_SAVE_PATH / sim_exp_name
            os.makedirs(sim_path.parent, exist_ok=True)
            with open(sim_path / f'{self.rnn_type}{additional_name}_seed{sim_seed}.json', 'w') as f:
                config['model_path'] = str(config['model_path'])
                json.dump(config, f, indent=4)
            joblib.dump(behav, sim_path / f'{self.rnn_type}{additional_name}_seed{sim_seed}.pkl')
        return behav

def _compare_score_to_target(self, input, target, mask, h0=None):
    pass

def _tensor_structure_to_numpy(obj):
    """Convert all tensors (nested) in best_model_pass to numpy arrays."""
    if isinstance(obj, dict):
        obj_new = {k: _tensor_structure_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        obj_new = [_tensor_structure_to_numpy(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        obj_new = obj.detach().cpu().numpy()
        if obj_new.size == 1:  # transform 1-element array to scalar
            obj_new = obj_new.item()
    else:
        obj_new = obj
    return obj_new