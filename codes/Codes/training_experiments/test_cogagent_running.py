"""
Test the interactions of agents and datasets.
"""
import sys
import numpy as np
sys.path.append('..')
from utils import goto_root_dir
from datasets import Dataset
from agents import Agent
from pathlib import Path
from training_experiments import training
from path_settings import *
from pathlib import Path


goto_root_dir.run()
# config = {
#       'dataset': 'BartoloMonkey',
#       'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)},
#       'behav_format': 'cog_session',
#       ### model info
#       'agent_type': 'PRLCog',
#       'cog_type': 'MB0',
#       'seed': 0,
#       'model_path': 'test_cogagent_running/test_cog_agent',
#       }
# config = {
#       'dataset': 'MillerRat',
#       'behav_data_spec': {'animal_name': 'm55'},
#       'behav_format': 'cog_session',
#       ### model info
#       'agent_type': 'RTSCog',
#       'cog_type': 'MB0',
#       'seed': 0,
#       'model_path': 'test_cogagent_running/test_cog_agent',
#       }
# config = {
#       'dataset': 'AkamRat',
#       'behav_data_spec': {'animal_name': 267},
#       'behav_format': 'cog_session',
#       ### model info
#       'agent_type': 'NTSCog',
#       'cog_type': 'MF_MB_bs_rb_ck',
#       'seed': 0,
#       'model_path': 'test_cogagent_running/test_cog_agent',
#       }

for sub in [0,29,1,30]:
      config = {
            ### dataset info
            'dataset': 'CPBHuman',
            'behav_format': 'cog_session',
            'behav_data_spec': {'subjects':sub, 'max_segment_length':30},

            # 'both' for all blocks
            ### model info
            'agent_type': 'CPBCog',
            'cog_type': 'MF',
            'seed': 0,
            'model_path': 'test_cogagent_running/test_CPBcog_agent',
      }
      behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
      behav_dt = behav_dt.behav_to(config)  # transform format following specifications
      print('Data block num', behav_dt.batch_size)
      behav_data = behav_dt.get_behav_data(np.arange(behav_dt.batch_size), {'behav_format': 'cog_session'})
      ag = Agent(config['agent_type'], config=config)
      # ag.save(verbose=True)
      # ag.load(config['model_path'])
      par_list = np.linspace(1,20,11)
      loss_list = []
      for par in par_list:
            ag.set_params([par])
            output = ag(behav_data['input'])
            # print(output.keys(), len(output['output']),len(output['internal']))
            loss_list.append(output['behav_loss'])

      from matplotlib import pyplot as plt
      plt.plot(par_list, loss_list, label=sub)
      plt.xlabel('parameter')
      plt.ylabel('loss')
      plt.legend()
      plt.show()
