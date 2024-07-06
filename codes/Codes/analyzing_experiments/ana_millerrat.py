from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

analyzing_pipeline = [
    'analyze_model_perf_for_each_exp',
    # 'compile_perf_for_all_exps',
    # 'extract_model_par',
    # 'run_scores_for_each_exp_best_for_test',
    # 'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'extract_ev_for_each_exp',
    # 'sym_regression_for_each_model',
    # 'neural_decoding_for_each_model',
    # 'analyze_model_perf_for_each_data_proportion',
]
exp_folders = [
    'exp_seg_millerrat55',
    'exp_seg_millerrat64',
    'exp_seg_millerrat70',
    'exp_seg_millerrat71',
]

## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                                 additional_rnn_keys={'model_identifier_keys': ['symm','finetune']})

if 'compile_perf_for_all_exps' in analyzing_pipeline:
    compile_perf_for_exps(exp_folders, 'exp_seg_millerrat',
                          rnn_filter={'readout_FC': True,
                                      'symm': 'none'}
                          )


if 'extract_model_par' in analyzing_pipeline:
        extract_model_par(exp_folders[0])

if 'extract_ev_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_sv_for_exp(exp_folder)
# dynamics
for exp_folder in exp_folders:
    if 'run_scores_for_each_exp' in analyzing_pipeline:
        run_scores_exp(exp_folder)
    if 'run_scores_for_each_exp_best_for_test' in analyzing_pipeline:
        run_scores_exp(exp_folder, best_for_test=True)
    if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
        run_2d_inits_exp(exp_folder, grid_num=50)


# 1d logit features
if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

#sym regression
if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_seg_millerrat55\rnn_type-SGRU.hidden_dim-2.readout_FC-False.l1_weight-0.0001\outerfold0_innerfold8_seed1'
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)

def extract_feature_func(kept_feat_idx):
    n_neurons, n_events, bin_num = kept_feat_idx.shape
    # print('kept_feat_idx', kept_feat_idx.shape, np.sum(kept_feat_idx))

    subset_from_kept_feat = []
    for n in range(n_neurons):
        for e in range(n_events):
            for bin in range(bin_num):
                if kept_feat_idx[n, e, bin]:
                    if (e == 2 and 10<=bin<=20) or (e == 3 and 5<=bin<15):
                        #if e in [2, 3] and 5<=bin<15:
                        subset_from_kept_feat.append(True)
                    else:
                        subset_from_kept_feat.append(False)
    subset_from_kept_feat = np.array(subset_from_kept_feat)
    return subset_from_kept_feat


# neural decoding; should be run after dynamics/scores analysis
if 'neural_decoding_for_each_model' in analyzing_pipeline:
    for exp_folder in exp_folders:
        neuro_data_spec={
            'start_time_before_event': -2,
            'end_time_after_event': 4,
            'bin_size': 0.2,
        }
        # run_decoding_exp(exp_folder, neuro_data_spec,
        #                  analyses=[
        #                      # 'decode_logit',
        #                      # 'decode_value',
        #                      # 'decode_chosen_value',
        #                      'value_decode_neuron',
        #                      'task_var_decode_neuron',
        #                      'task_var_value_decode_neuron',
        #                  ])
        compile_decoding_results(exp_folder, neuro_data_spec, extract_feature_func=extract_feature_func)


exp_folders = [
    'exp_seg_millerrat55_dataprop',
    'exp_seg_millerrat64_dataprop',
    'exp_seg_millerrat70_dataprop',
]
if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
                             additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )
    # compile_perf_for_exps(exp_folders, 'exp_seg_millerrat_dataprop',
    #                       additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
    #                       additional_cog_keys={'model_identifier_keys':['trainval_percent']},
    #                       additional_rnn_agg={'trainval_percent': ('trainval_percent','max')},
    #                       additional_cog_agg={'trainval_percent': ('trainval_percent','max')})
