from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
import re
goto_root_dir.run()

analyzing_pipeline = [
    # 'analyze_model_perf_for_each_exp',
    # 'run_scores_for_each_exp',
    'run_itself_scores_for_each_exp',
]
exp_folders = [
    # 'exp_metatwostep',
    # 'exp_metatwostep_seed0',
    # 'exp_metatwostep_seed0_savedpoint1050',
    # 'exp_metatwostep_seed0_savedpoint1100',
    # 'exp_metatwostep_seed0_savedpoint1150',
    # 'exp_metatwostep_seed0_savedpoint1200',
    # 'exp_metatwostep_seed0_savedpoint1250',
    # 'exp_metatwostep_seed1',
    # 'exp_metatwostep_seed2',
    # 'exp_metatwostep_seed3',
    # 'exp_metatwostep_seed4',
    'exp_metatwostep_seed4_savedpoint1050',
    'exp_metatwostep_seed4_savedpoint1100',
    'exp_metatwostep_seed4_savedpoint1150',
    'exp_metatwostep_seed4_savedpoint1200',
    'exp_metatwostep_seed4_savedpoint1250',

]

## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'RTSCog', has_rnn=False)

# dynamics
def run_itself_scores_exp(exp_folder):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    # if os.path.exists(ana_exp_path / 'total_scores.pkl'):
    #     print('already exists')
    #     return
    if 'savedpoint' in exp_folder:
        # using re to find "savedpoint?"
        savedpoint = int(re.findall(r'savedpoint(\d+)', exp_folder)[0])
    else:
        savedpoint = -1
    if 'seed' in exp_folder:
        seed = int(re.findall(r'seed(\d+)', exp_folder)[0])
        dt = Dataset('MetaTwoStep', {'seed': seed, 'savedpoint': savedpoint})
    else:
        dt = Dataset('MetaTwoStep', {'savedpoint': savedpoint})
    model_LS_logits = dt.neuro['LS_policy_logits'] # shape: episode_num, trial_num
    model_scores = dt.neuro['policy_logits'][:,:, 1,1:] # policy_logits shape: episode_num, trial_num, 3, 3=(F, L, R)
    model_internal = dt.neuro['activity'][:,:,1,:] # activity shape: episode_num, trial_num, 3, neuron_num
    os.makedirs(ana_exp_path / 'metaRL', exist_ok=True)
    os.makedirs(ana_exp_path / 'metaRL_LS', exist_ok=True)
    # model_scores: 1000
    # trial_type: 1000
    # model_LS_logits: 1001
    joblib.dump({
        'scores': model_scores,
        'internal': model_internal,
        'trial_type': dt.behav['trial_type'][:,:-1],
        'hid_state_lb': np.zeros(0),
        'hid_state_ub': np.zeros(0),
    }, ana_exp_path / 'metaRL' / f'total_scores.pkl')

    joblib.dump({
        'scores': model_LS_logits,
        'internal': model_LS_logits,
        'trial_type': dt.behav['trial_type'],
        'hid_state_lb': np.zeros(0),
        'hid_state_ub': np.zeros(0),
    }, ana_exp_path / 'metaRL_LS' / f'total_scores.pkl')

if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder)

if 'run_itself_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_itself_scores_exp(exp_folder)
