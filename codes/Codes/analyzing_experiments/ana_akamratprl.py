from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    # 'compile_perf_for_all_exps',
    # 'extract_model_par',
    # 'logit_vs_action_freq',
    # 'action_freq_after_action_seq',
    # 'run_scores_for_each_exp',
    # 'run_scores_for_each_exp_best_for_test',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
]

exp_folders = [
    'exp_seg_akamratprl388',
    'exp_seg_akamratprl383',
    'exp_seg_akamratprl382',
    'exp_seg_akamratprl380',
    'exp_seg_akamratprl368',
    'exp_seg_akamratprl367',
    'exp_seg_akamratprl361',
    'exp_seg_akamratprl360',
    'exp_seg_akamratprl359',
    'exp_seg_akamratprl358',
]

# perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                                 additional_rnn_keys={'model_identifier_keys': ['complex_readout','finetune','symm']}
                                 )

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_seg_akamratprl',
                          rnn_filter={'readout_FC': True})

if 'extract_model_par' in analyzing_pipeline:
    extract_model_par(exp_folders[0])

if 'logit_vs_action_freq' in analyzing_pipeline:
    for exp_folder in exp_folders[:1]:
        logit_vs_action_freq(exp_folder)

if 'action_freq_after_action_seq' in analyzing_pipeline:
    for exp_folder in exp_folders:
        action_freq_after_action_seq(exp_folder)
# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder)
    if 'run_scores_for_each_exp_best_for_test' in analyzing_pipeline:
        run_scores_exp(exp_folder, best_for_test=True)
    if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
        run_2d_inits_exp(exp_folder, grid_num=50)


if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

