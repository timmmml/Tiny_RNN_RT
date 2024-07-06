import os

from datasets import Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import goto_root_dir
import joblib
from path_settings import *

goto_root_dir.run()
exp_folder = 'RTS_agents_millerrat55'
dts = {
    ag: Dataset('SimAgent', behav_data_spec={'agent_path': exp_folder,'agent_name': ag})
    for ag in [
        'MB0s_seed0',
        'LS0_seed0',
        'LS1_seed0',
        'MB0_seed0',
        'MB1_seed0',
        # 'MB0md_seed0',
        'RC_seed0',
        'Q(0)_seed0',
        'Q(1)_seed0'
    ]
}
dts |= {'m55': Dataset('MillerRat', behav_data_spec={'animal_name': 'm55'})}
for dt_name, dt in dts.items():
    lag = 10
    X_list = []
    y_list = []
    predictor_included = ['reward-common', 'reward-rare', 'nonreward-common', 'nonreward-rare']
    predictor_num = len(predictor_included)
    for episode in range(dt.batch_size):
        action = dt.behav['action'][episode]
        stage2 = dt.behav['stage2'][episode]
        reward = dt.behav['reward'][episode]
        trial_num = len(action)
        common = (action == stage2) * 1
        pred_outcome = reward - 0.5 # +0.5 for rewarded trials, -0.5 for not rewarded trials
        pred_transition = common - 0.5 # +0.5 for common transition, -0.5 for rare transition
        pred_interact = pred_outcome * pred_transition * 2
        # +0.5 for common transition rewarded and rare transitions non-rewarded trials
        # -0.5 for rare transition rewarded and common transition non-rewarded trials
        predictors = {
            'outcome': pred_outcome,
            'transition': pred_transition,
            'interaction': pred_interact,
            'reward-common': reward * common * 0.5,
            'reward-rare': reward * (1 - common) * 0.5,
            'nonreward-common': (1 - reward) * common * 0.5,
            'nonreward-rare': (1 - reward) * (1 - common) * 0.5,
        }
        X = np.zeros([trial_num - lag, lag, predictor_num])
        y = np.zeros([trial_num - lag])
        for i in range(trial_num - lag):
            y[i] = (action[i + lag] == action[i + lag - 1]) * 1
            for l in range(lag):
                for j, pred in enumerate(predictor_included):
                    X[i, l, j] = predictors[pred][i + l]
        X_list.append(X)
        y_list.append(y)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    total_trial_num = X.shape[0]
    print(X.shape, y.shape)
    clf = LogisticRegression(random_state=0).fit(X.reshape([total_trial_num, lag * predictor_num]), y)
    coef = clf.coef_.reshape([lag, predictor_num])
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    filename = '.'.join(predictor_included)
    filename = f'{dt_name}.lag{lag}.{filename}.pkl'
    os.makedirs(ana_exp_path, exist_ok=True)
    joblib.dump({
        'predictor_included': predictor_included,
        'lag': lag,
        'coef': coef,
    }, ana_exp_path / filename)
