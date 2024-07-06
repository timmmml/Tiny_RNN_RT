from .BaseAgent import BaseAgent
import os
import json
import joblib
from path_settings import *
import numpy as np
import random

class CogAgent(BaseAgent):
    """All types of Cog agents.

    Currently all Cog agents come from Akam's two-step task and novel two-step task.

    Attributes:
        model: the Cog agent, implemented in Akam's way.

    """

    def __init__(self, config=None):
        super().__init__()
        if not hasattr(self, 'config'):
            self.config = config
            self.model = None
            self.params = None
            self.cog_type = None
            self.state_vars = []

    def load(self, model_path, strict=True):
        """Load model parameters from disk.
        Args:
            strict: not used.
        """
        self.set_params(joblib.load(MODEL_SAVE_PATH / model_path / 'model.ckpt'))

    def save(self, params=None, verbose=False):
        """Save config, model, and results."""
        model_path = self.config['model_path']
        os.makedirs(MODEL_SAVE_PATH / model_path, exist_ok=True)
        # save config
        self.save_config()

        # save model parameters
        if params is None: # current state dict is saved
            joblib.dump(self.params, MODEL_SAVE_PATH / model_path / 'model.ckpt')
        else:
            joblib.dump(params, MODEL_SAVE_PATH / model_path / 'model.ckpt')


        if verbose: print('Saved model at', MODEL_SAVE_PATH / model_path)
        pass

    def forward(self, input, h0=None, standard_output=True):
        """Process a batch of input.

        Args:
            input: List of nn_session instances, each of which contains a block of trials.
            h0: initial hidden state
        Returns:
            a dict of internal states and outputs
        """
        if isinstance(input, dict):
            input = input['input']
        nn_sessions = input
        params = self.model.params
        internal = {}
        scores = []
        behav_loss_sessions = []
        total_behav_loss = 0
        total_trial_num = 0
        for s in nn_sessions:
            if h0 is not None:
                self.model.init_wm(wm={'h0': h0}, params=params)
            internal_session = self.model.session_likelihood(s, params, get_DVs=True)
            total_trial_num += s.n_trials if hasattr(s, 'n_trials') else s['n_trials']
            total_behav_loss += -internal_session['session_log_likelihood']
            scores.append(internal_session['scores'])
            behav_loss_sessions.append(internal_session['session_log_likelihood'])
            for k, v in internal_session.items():
                if k not in ['scores', 'session_log_likelihood']:
                    internal.setdefault(k, []).append(v)
            state_var_concat = []
            if hasattr(self.model, 'state_vars'):
                for k in self.state_vars:
                    v = internal_session[k]
                    match len(v.shape):
                        case 1:
                            v = v.reshape(v.shape[0], 1)
                        case 2:
                            pass
                        case _:
                            v = v.reshape(v.shape[0], -1)

                    state_var_concat.append(v)
            if len(state_var_concat) > 0:
                state_var = np.concatenate(state_var_concat, axis=1)
            else:
                state_var = np.array([])
            internal.setdefault('state_var', []).append(state_var)
        return {'output': scores, 'internal': internal,
                'behav_loss': total_behav_loss / total_trial_num, 'total_trial_num': total_trial_num}


    def _set_init_params(self):
        """Set initial parameters for the model.

        This is a helper function. self.model should at least has the param_range attribute, which is a list of range strings.
        Range strings:
            'unit':  # Range: 0 - 1.
            'half':  # Range: 0 - 0.5
            'pos':  # Range: 0 - inf
            'unc': # Range: - inf - inf.
        """

        if hasattr(self.model, 'params'):
            # for some agents, the initial parameters are already set
            self.params = self.model.params
        else:
            # for some agents, the initial parameters are not set; inferred them from params_ranges
            self.model.params = []
            for param_range in self.model.param_ranges:
                self.model.params.append({'unit': 0.5, 'pos': 5, 'unc': 0.2, 'half': 0.2}[param_range])
            self.params = self.model.params
        self.num_params = len(self.params)

    def set_params(self, params):
        """Set model parameters."""
        for i, param in enumerate(params):
            self.model.params[i] = param # make the change in-place
        self.params = self.model.params


    def simulate(self, task, config, save=True):
        """Simulate the agent's behavior on the task.

        Parameters should be set before this function; otherwise, the default parameters will be used.

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
        print('Simulating cog agent', self.cog_type, 'with params', self.model.params)
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
        behav['params'] = self.model.params
        behav['mid_vars'] = []

        np.random.seed(sim_seed)
        random.seed(sim_seed)

        for _ in range(n_blocks):
            self.DVs = self.model.simulate(task, n_trials, get_DVs=True)
            behav['action'].append(self.DVs['choices'])
            behav['stage2'].append(self.DVs['second_steps'])
            behav['reward'].append(self.DVs['outcomes'])
            for k in ['choices', 'second_steps', 'outcomes']:
                self.DVs.pop(k)
            behav['mid_vars'].append(self.DVs)
        if save:
            if len(additional_name) and additional_name[0] != '_':
                additional_name = '_' + additional_name
            sim_path = SIM_SAVE_PATH / sim_exp_name
            os.makedirs(sim_path, exist_ok=True)
            with open(sim_path / f'{self.cog_type}{additional_name}_seed{sim_seed}.json', 'w') as f:
                config['model_path'] = str(config['model_path'])
                json.dump(config, f, indent=4)
            joblib.dump(behav, sim_path / f'{self.cog_type}{additional_name}_seed{sim_seed}.pkl')
        return behav


def choose(P):
    "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
    return sum(np.cumsum(P) < random())