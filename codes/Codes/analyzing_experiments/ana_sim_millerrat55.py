from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'sym_regression_for_each_model',
]

exp_folders = [
    'exp_sim_millerrat55',
    # 'exp_sim_millerrat55_nblocks200',
]


## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                             additional_rnn_keys={'model_identifier_keys': ['agent_name']},
                             additional_cog_keys={'model_identifier_keys': ['agent_name']})


# dynamics
if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder)

if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_2d_inits_exp(exp_folder, grid_num=50)

#sym regression
# if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_sim_millerrat55\agent_name-LS0_seed0.rnn_type-SGRU.hidden_dim-1.readout_FC-True.l1_weight-1e-05\outerfold6_innerfold3_seed0',
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)