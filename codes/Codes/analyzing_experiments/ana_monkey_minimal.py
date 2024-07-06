from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    'compile_perf_for_all_exps',
    'run_scores_for_each_exp',
    'run_2d_inits_for_each_exp',
    'extract_1d_for_each_exp',
]

exp_folders = [
    'exp_monkeyV_minimal',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                                 additional_rnn_keys={'model_identifier_keys': ['complex_readout','symm','finetune']}
                                 )

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_monkey',
                          rnn_filter={'readout_FC': True,
                                      'symm': 'none',
                                      'complex_readout': 'none'}
                          )

# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder)
    if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
        run_2d_inits_exp(exp_folder, grid_num=50)

if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)
