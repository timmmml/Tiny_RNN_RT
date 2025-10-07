"""
Run all models on monkey W
Vary the data proportion.
"""
import sys
sys.path.append('..')
from training_experiments.training import *
from training_experiments.training_Nautilus_jobs_generation import *

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': 'W', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
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
      'trainval_percent': 100, # subsample the training-validation data (percentage)
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

trainval_percent_list = [100, 90, 80, 70, 60, 50, 40, 30, 20]#, 10]

config_ranges = { # keys are used to generate model names
      'rnn_type': ['GRU'],
      'hidden_dim': [2],
      'readout_FC': [True],
      'l1_weight': [1e-4, 1e-3, 1e-2, 1e-1],
      'trainval_percent': trainval_percent_list,
}

resource_dict = {'memory': 5, 'cpu': 16, 'gpu': 0}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

base_config.update({
      'input_dim': 4,
      'one_hot': True,
})

config_ranges.update({
      'rnn_type': ['SGRU'],
})

# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

######## change to inner_split=2
base_config.update({
      'input_dim': 3,
      'one_hot': False,
})
config_ranges.update({
      'rnn_type': ['GRU'],
      'l1_weight': [1e-4, 1e-3, 1e-2, 1e-1],
      'trainval_percent': [18, 9], # equal to 10% and 5% of the original data with inner_split=2
      'inner_splits': [2],
})
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)

base_config.update({
      'input_dim': 4,
      'one_hot': True,
})
config_ranges.update({
      'rnn_type': ['SGRU'],
})
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict)
######## end of change to inner_split=2

base_config = {
      ### dataset info
      'dataset': 'BartoloMonkey',
      'behav_format': 'cog_session',
      'behav_data_spec': {'animal_name': 'W', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
      # 'both' for all blocks
      ### model info
      'agent_type': 'PRLCog',
      'cog_type': 'MB0',
      'device': 'cpu',
      ### training info for one model
      ### training info for many models on dataset
      'trainval_percent': 100, # subsample the training-validation data (percentage)
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'minimal', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
}

config_ranges = {  # keys are also used to generate model names
      'cog_type': [
            # 'BAS',
            'RC', 'MB0', 'MB0s', 'MB0se', 'MB0md', 'MB0m', 'MB1', 'LS0', 'LS1', 'Q(0)',
            # 'MB0mdnb',
      ],
      'trainval_percent': trainval_percent_list,
}

# resource_dict = {'memory': 10, 'cpu': 16, 'gpu': 0}
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, n_jobs=-1)

if __name__ ==  '__main__' or '.' in __name__:
      behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)

######## change to inner_split=2
config_ranges.update({
        'trainval_percent': [18, 9], # equal to 10% and 5% of the original data with inner_split=2
        'inner_splits': [2],
})
# behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, n_jobs=-1)

######## end of change to inner_split=2
# if __name__ ==  '__main__' or '.' in __name__:
#       behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=-1, verbose_level=1)