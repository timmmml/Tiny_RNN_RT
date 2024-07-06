import pandas as pd

from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from matplotlib import pyplot as plt
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    # 'run_scores_for_each_exp',
]

exp_folders = [
    'exp_seg_akamrat49_distill',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'NTSCog',
                                 additional_rnn_keys={'model_identifier_keys': ['include_embedding','embedding_dim','trainval_size','distill','teacher_prop']},
                                 additional_cog_keys={'model_identifier_keys':['trainval_size','distill']}
                                 )
        # ana_exp_path = ANA_SAVE_PATH / exp_folder
        # cog_perf_all = joblib.load(ana_exp_path / f'cog_final_perf.pkl')
        # for cog_type in pd.unique(cog_perf_all['cog_type']):
        #     cog_perf = cog_perf_all[cog_perf_all['cog_type'] == cog_type]
		#
        #     if cog_perf[cog_perf['subjects'] == -1].shape[0] > 0:
        #         print('already exists')
        #         continue
        #     row = cog_perf.iloc[0]
        #     filter = cog_perf['total_test_loss'] < 2000 # 1 subject has a very high loss
        #     total_test_loss = cog_perf[filter]['total_test_loss'].sum()
        #     test_trial_num = cog_perf[filter]['test_trial_num'].sum()
        #     avg_loss = total_test_loss / test_trial_num
        #     print(f'avg loss: {avg_loss}')
        #     new_row= {
        #         'cog_type': row['cog_type'],
        #         'subjects': -1, # all subjects
        #         'total_test_loss': total_test_loss,
        #         'test_trial_num': test_trial_num,
        #         'test_loss': avg_loss,
        #         'hidden_dim': row['hidden_dim'],
        #     }
        #     cog_perf_all = pd.concat([cog_perf_all, pd.DataFrame([new_row])], ignore_index=True)
        # print(cog_perf_all)
        # joblib.dump(cog_perf_all, ana_exp_path / f'cog_final_perf.pkl')

# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder, model_filter={'distill': 'teacher',
                                                 'hidden_dim': 20,
                                                 'embedding_dim': 8,
                                                 },
                       overwrite_config={
                           'behav_data_spec': ['animal_name', 'max_segment_length', 'include_embedding','augment'],
                           'augment': True}
                       )
