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
    # 'analyze_rt_for_each_exp',
    # 'extract_1d_for_each_exp',
    # 'sym_regression_for_each_model',
    # 'neural_decoding_for_each_model',
    # 'neural_decoding_two_model_compare',
    # 'analyze_model_perf_for_each_data_proportion',
    # 'analyze_markov_matrix_for_each_exp',
]

exp_folders = [
    # 'exp_monkeyV',
    'exp_monkeyW',
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

if 'analyze_rt_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        compare_logit_rt(exp_folder)

if 'extract_1d_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        extract_1d_logit_for_exp(exp_folder)

#sym regression
if 'sym_regression_for_each_model' in analyzing_pipeline:
    model_paths = [
        r'exp_monkeyV\rnn_type-GRU.hidden_dim-2.readout_FC-True.l1_weight-0.0001\outerfold0_innerfold8_seed0',
    ]
    for model_path in model_paths:
        sym_regression_for_model(model_path)


def extract_feature_func(kept_feat_idx):
    neuron_num, bin_num = kept_feat_idx.shape
    # print(session_name, 'kept_feat_idx', kept_feat_idx.shape, np.sum(kept_feat_idx))
    subset_from_kept_feat = []
    selected_bins = [2,3,4]
    for n in range(neuron_num):
        for bin in range(bin_num):
            if kept_feat_idx[n, bin]:
                if bin in selected_bins:
                    subset_from_kept_feat.append(True)
                else:
                    subset_from_kept_feat.append(False)
    subset_from_kept_feat = np.array(subset_from_kept_feat)
    return subset_from_kept_feat

# neural decoding; should be run after dynamics/scores analysis
if 'neural_decoding_for_each_model' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_decoding_exp(exp_folder, {'block_type': 'where'},
                         analyses=[
                             # 'decode_logit',
                             'decode_logit_change',
                             # 'decode_value',
                             # 'decode_chosen_value',
                         ])
        # run_decoding_exp(exp_folder, {'block_type': 'where'},
        #                  analyses=[
        #                      # 'value_decode_neuron',
        #                      # 'task_var_decode_neuron',
        #                      # 'task_var_value_decode_neuron',
        #                  ])
        ## compile_decoding_results only used when use variables to decode neuronal activity
        # compile_decoding_results(exp_folder, {'block_type': 'where'}, extract_feature_func=extract_feature_func)

if 'neural_decoding_two_model_compare' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_two_model_compare_decoding(exp_folder, {'block_type': 'where'})

exp_folders = [
    'exp_monkeyV_dataprop',
    'exp_monkeyW_dataprop',
]
if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
                             additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )


def create_transition_graph(d):
    import graphviz
    g = graphviz.Digraph('transition_graph', format='png', engine='neato')

    # Nodes with pos attribute
    g.node('00', 'A1 R: 0', shape='rectangle', pos='0,3!')
    g.node('01', 'A1 R: 1', shape='rectangle', pos='0,0!')
    g.node('10', 'A2 R: 0', shape='rectangle', pos='4,3!')
    g.node('11', 'A2 R: 1', shape='rectangle', pos='4,0!')
    g.node('0', 'A1', shape='circle', pos='2,3!')
    g.node('1', 'A2', shape='circle', pos='2,0!')

    for (action, reward), next_actions in d.items():
        for next_action, probability in next_actions.items():
            intermediate_node = f"{action}{reward}"
            g.edge(intermediate_node, str(next_action), label=f"{probability:.2f}")

    return g

def analyze_markov_matrix_for_each_exp(animal_name):
    config = {
        ### dataset info
        'dataset': 'BartoloMonkey',
        'behav_format': 'tensor',
        'behav_data_spec': {'animal_name': animal_name, 'filter_block_type': 'both', 'block_truncation': (10, 70)},
    }
    behav_data_spec = config['behav_data_spec']
    dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec)
    from collections import defaultdict
    transition_count = defaultdict(lambda: defaultdict(int))
    for block in range(len(dt.behav['action'])):
        action = dt.behav['action'][block]
        reward = dt.behav['reward'][block]
        for t in range(len(action)-1):
            transition_count[(action[t], reward[t])][action[t+1]] += 1.0
    # to frequency

    for k, v in transition_count.items():
        total = sum(v.values())
        for k2, v2 in v.items():
            transition_count[k][k2] = v2/total
    # change to normal dict of dict
    transition_count = {k: dict(v) for k, v in transition_count.items()}
    print(transition_count)
    transition_graph = create_transition_graph(transition_count)
    transition_graph.render(f'transition_graph_{animal_name}', view=True)

if 'analyze_markov_matrix_for_each_exp' in analyzing_pipeline:
    analyze_markov_matrix_for_each_exp('V')