from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    # 'compile_perf_for_all_exps',
    # 'extract_model_par',
    # 'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'run_scores_for_each_exp_best_for_test',
    # 'extract_ev_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'compile_1d_for_all_exps',
    # 'run_2d_inits_for_each_exp',
    'analyze_model_perf_for_each_data_proportion',
]

exp_folders = [
'exp_seg_akamrat49',
'exp_seg_akamrat50', # missing a few models
'exp_seg_akamrat51',
'exp_seg_akamrat52',
'exp_seg_akamrat53',
'exp_seg_akamrat54',
'exp_seg_akamrat95',
'exp_seg_akamrat96',
'exp_seg_akamrat97',
'exp_seg_akamrat98',
'exp_seg_akamrat99',
'exp_seg_akamrat100',
'exp_seg_akamrat264',
'exp_seg_akamrat268',
'exp_seg_akamrat263',
'exp_seg_akamrat266',
'exp_seg_akamrat267',
# 'exp_seg_akamratAll',
]
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'NTSCog',
                                 additional_rnn_keys={'model_identifier_keys': ['symm','finetune']})

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_seg_akamrat',
                          rnn_filter={'readout_FC': True,
                                      'symm': 'none'})


if 'extract_model_par' in analyzing_pipeline:
        extract_model_par(exp_folders[0])

if 'extract_ev_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_sv_for_exp(exp_folder)

# dynamics
if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder)
if 'run_scores_for_each_exp_best_for_test' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder, best_for_test=True)
if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_2d_inits_exp(exp_folder, grid_num=50)

# 1d logit features
if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

if 'compile_1d_for_all_exps' in analyzing_pipeline:
    compile_1d_logit_for_exps(exp_folders, 'exp_seg_akamrat')

# exp_folders = [
# 'exp_nonseg_akamrat49',
# 'exp_nonseg_akamrat50',
# 'exp_nonseg_akamrat51',
# 'exp_nonseg_akamrat52',
# 'exp_nonseg_akamrat53',
# 'exp_nonseg_akamrat54',
# ]
# for exp_folder in exp_folders:
#     find_best_models_for_exp(exp_folder, 'NTSCog')

exp_folders = [
        'exp_seg_akamrat49_dataprop',
]
if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'NTSCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent', 'trainprob']},
                             additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )