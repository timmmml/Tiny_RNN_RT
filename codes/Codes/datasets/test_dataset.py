""" Test whether these datasets work well. """
import pandas as pd
from analyzing_experiments.analyzing import pd_full_print_context
from datasets import Dataset
import numpy as np
from pathlib import Path
from utils import goto_root_dir
from path_settings import *
import joblib
if __name__ == '__main__':
    goto_root_dir.run()

    # dt = Dataset('BartoloMonkey',
    #         behav_data_spec={'animal_name': 'all', 'filter_block_type': 'both', 'block_truncation': (10, 70)}).behav_to({'behav_format':'tensor'})
    # for rat in ['m71','m70', 'm64','m55']:
    #     dt = Dataset('MillerRat',
    #             behav_data_spec={'animal_name': rat, #'m64',#'m55'
    #                              'max_segment_length': 150,
    #                              },
    #             neuro_data_spec={'start_time_before_event': -2,
    #                                 'end_time_after_event': 4,
    #                  'bin_size': 0.2,}
    #                  ).behav_to({'behav_format':'tensor'})
    #     print(rat, dt.total_trial_num)
    # dt = Dataset('SimAgent',
    #              behav_data_spec={'agent_path': ['RTS_agents_millerrat55', 'LS0_seed0']
    #                               }).behav_to({'behav_format': 'tensor'})
    # total_trial_num = 0
    # for rat in ['49', '50', '51', '52', '53', '54', '100', '95', '96', '97', '98', '99', '263', '264', '266', '267', '268']:
    #     dt = Dataset('AkamRat',
    #                  behav_data_spec={'animal_name': rat, 'max_segment_length': 150,
    #                                   }).behav_to({'behav_format': 'tensor'})
    #     print(rat, dt.total_trial_num)
    #     total_trial_num += dt.total_trial_num
    # print(total_trial_num)
    dt = Dataset('AkamRat',
                 behav_data_spec={'animal_name': 'all', 'max_segment_length': 150, 'include_embedding': True, 'augment': True,
                                  }).behav_to({'behav_format': 'tensor', 'include_embedding': True,})
    block_idx_after_aug = dt.get_after_augmented_block_number([0,1,2,3])
    # joblib.dump(dt.data_summary(), ANA_SAVE_PATH / 'AkamRat' / 'subject_summary.pkl')
    # dt = Dataset('AkamRat',
    #         behav_data_spec={'animal_name': 358, 'max_segment_length': 150,
    #                         'task': 'reversal_learning', }).behav_to({'behav_format':'tensor'})
    # for rat in [269, 270, 271, 272, 273, 274, 275, 277, 278, 279]:
    #     dt = Dataset('AkamRat',
    #                  behav_data_spec={'animal_name': rat, 'max_segment_length': 150,
    #                                   'task': 'no_transition_reversal', }).behav_to({'behav_format': 'tensor'})
    #     print(rat, dt.total_trial_num)

    # joblib.dump(dt.data_summary(), ANA_SAVE_PATH / 'AkamRatRTS' / 'subject_summary.pkl')
    # subject_summary = dt.data_summary()
    # with pd_full_print_context():
    #     print(subject_summary)

    # dt = Dataset('DezfouliHuman',behav_data_spec={'subject_number': 'all', 'include_embedding': True}).behav_to({'behav_format': 'tensor','include_embedding': True})

    # dt = Dataset('CPBHuman',
    #                      behav_data_spec={
    #                                       'max_segment_length': 30,
    #                                       'input_noss': False,
    #                                     'include_embedding': True,
    #                                       }).behav_to({'behav_format':'tensor','include_embedding': True,})
    # print(dt.batch_size)
    # behav_dt = dt.get_behav_data(list(range(dt.batch_size)), {'behav_format':'tensor'})
    # for session_name in ['V20161005', 'V20160929', 'V20160930', 'V20161017']:
    #     neuro_dt = dt.get_neuro_data(session_name=session_name, block_type='where', zcore=True, remove_nan=True, shape=2)[0]
    #     print(neuro_dt.shape)

    # for session_name in ['2015-05-13','2015-05-14','2015-05-19','2015-05-26']:
    #     neuro_dt = dt.get_neuro_data(session_name=session_name, zcore=True, remove_nan=True, shape=2)[0]
    #     print(neuro_dt.shape)

    # from datasets.LaiHumanDataset import LaiHumanDataset
    # data_path = "D:\\OneDrive\\Documents\\git_repo\\cognitive_dynamics\\files"
    # dt = LaiHumanDataset(data_path, behav_data_spec={'group': [0, 1]}).behav_to({'behav_format': 'cog_session',})
    # behav_dt = dt.get_behav_data(list(range(dt.batch_size)), {'behav_format': 'cog_session'})
    
    # dt = Dataset('LaiHuman',behav_data_spec={'group':[0,1]}).behav_to({'behav_format': 'tensor', 'include_cond': True, 'include_sub': True})
    # behav_dt = dt.get_behav_data(list(range(dt.batch_size)), {'behav_format':'tensor'})
    #
    # dt = Dataset('LaiHuman',behav_data_spec={'subjects':[84]}).behav_to({'behav_format': 'cog_session'})
    # behav_dt = dt.get_behav_data(list(range(dt.batch_size)), {'behav_format': 'cog_session'})

    # animal_enough_trials = []
    # for animal in ['IBL-T1',
    #         #        'IBL-T2', 'IBL-T3', 'IBL-T4', 'NYU-01', 'NYU-02', 'NYU-04', 'NYU-06',
    #         # 'CSHL_001', 'CSHL_002', 'CSHL_003', 'CSHL_005', 'CSHL_007', 'CSHL_008', 'CSHL_010', 'CSHL_012',
    #         # 'CSHL_014', 'CSHL_015', 'KS003', 'KS005', 'KS019', 'ZM_1367', 'ZM_1369', 'ZM_1371', 'ZM_1372',
    #         # 'ZM_1743', 'ZM_1745', 'ZM_1746', 'ibl_witten_04', 'ibl_witten_05', 'ibl_witten_06', 'ibl_witten_07',
    #         # 'ibl_witten_12', 'ibl_witten_13', 'ibl_witten_14', 'ibl_witten_15', 'ibl_witten_16'
    #                ]:
    #     dt = Dataset('IBL',behav_data_spec={'animal_name': animal,'output_h0': False}).behav_to({'behav_format': 'tensor'})
    #     if dt.batch_size > 45:
    #         animal_enough_trials.append(animal)
    #     behav_dt = dt.get_behav_data(list(range(dt.batch_size)), {'behav_format': 'tensor'})
    # print(animal_enough_trials)

    # dt = Dataset('MetaTwoStep', {})
