"""
Run all models on monkey V
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cpu',
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 2000,
      'early_stop_counter': 200,
      ### training info for many models on dataset
      'outer_splits': 5,
      'inner_splits': 4,
      'seed_num': 2,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [2],
      'readout_FC': [True],
      'l1_weight': [1e-5],
}

behavior_cv_training_config_combination(base_config, config_ranges)


base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'cog_session',
      'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      # 'both' for all blocks
      ### model info
      'agent_type': 'PRLCog',
      'cog_type': 'MB0',
      'device': 'cpu',
      ### training info for one model
      ### training info for many models on dataset
      'outer_splits': 5,
      'inner_splits': 4,
      'seed_num': 2,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = {  # keys are also used to generate model names
      'cog_type': [
            'MB0', 'MB1', 'LS0', 'Q(0)',
      ],
}
if __name__ ==  '__main__' or '.' in __name__:
      behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)