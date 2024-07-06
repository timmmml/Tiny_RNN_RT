import pandas as pd

from analyzing import *
from analyzing_dynamics import *
import os
from datasets import Dataset
import numpy as np
from utils import goto_root_dir
import joblib
from path_settings import *
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time


def extract_pcs(X):
    """ Extract PCs from X.

    Args:
        X: (n_samples, n_features)

    Returns:
        PC_result: dict
    """
    assert len(X.shape) == 2, X.shape
    n_samples, n_features = X.shape
    pca_num = min(n_samples, n_features)
    pca = PCA(pca_num)
    X = pca.fit_transform(X - X.mean(0))
    var_ratio = pca.explained_variance_ratio_
    var_cum = np.cumsum(var_ratio)
    part_ratio = var_ratio.sum()**2/(var_ratio**2).sum()
    return {
        'PC': X,
        'variance_ratio': var_ratio,
        'variance_cumulative_ratio': var_cum,
        'participation_ratio': part_ratio,
    }


def quick_decoding(X, y, alphas=None):
    """ Quick decoding using RidgeCV.
    This model used the validation set to estimate test loss. Not recommended.
    """
    raise NotImplementedError
    assert len(X.shape) == 2, X.shape
    assert len(y.shape) == 1, y.shape
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    clf = RidgeCV(alphas=alphas,store_cv_values=True)
    clf.fit(X, y)
    loc = np.where(alphas==clf.alpha_)[0][0] # find the index of the best alpha
    mse = clf.cv_values_[:, loc].mean() # mean cross-validation error for the best alpha
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean())
    r2 = 1 - mse/null_mse # cross-validated R^2
    return r2


def quick_decoding_multidim(X, y, alphas=None):
    """ Quick decoding using RidgeCV.
    This model used the validation set to estimate test loss. Not recommended.
    """
    raise NotImplementedError
    n_targets = y.shape[1]
    assert len(X.shape) == 2, X.shape
    assert len(y.shape) == 2, y.shape
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    clf = RidgeCV(alphas=alphas, store_cv_values=True, alpha_per_target=True)
    clf.fit(X, y)
    # clf.alpha_ shape: (n_targets,)
    # alphas shape: (n_alphas,)
    locs = clf.alpha_.reshape([-1, 1]) == alphas.reshape([1, -1]) # shape: (n_targets, n_alphas)
    target_locs, alpha_locs = np.where(locs)
    assert (target_locs == np.arange(n_targets)).all() # each target has a best alpha
    cv_errors = clf.cv_values_[:, target_locs, alpha_locs] # CV error when each target with the best alpha shape: (n_samples, n_targets)
    mse = cv_errors.mean(0) # mean cross-validation error over samples
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean(0).reshape([1, -1]), multioutput='raw_values')
    r2 = 1 - mse/null_mse # cross-validated R^2
    return r2


def decoding_CV(X, y, alphas=None):
    if len(y.shape) == 1:
        y = y.reshape([-1, 1])
    n_targets = y.shape[1]
    assert len(X.shape) == 2, X.shape
    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred = np.zeros(y.shape)
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RidgeCV(alphas=alphas, store_cv_values=True, alpha_per_target=True)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    mse = mean_squared_error(y, y_pred, multioutput='raw_values')
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean(0).reshape([1, -1]), multioutput='raw_values')
    r2 = 1 - mse/null_mse # cross-validated R^2
    return {'mse': mse, 'null_mse': null_mse, 'r2': r2}


def pca_decoding(X_PCs, y, verbose=True, max_pca_num=-1):
    """ Decoding y from X_PCs.

    Args:
        X_PCs: (n_samples, n_PCs), extracted PCs of X
        y: (n_samples,)
        verbose: bool
        max_pca_num: int, maximum number of PCs to use

    Returns:
        R2: (n_PCs,)
    """
    sample_num, feature_num = X_PCs.shape
    assert len(y.shape) == 1, y.shape
    if max_pca_num == -1:
        max_pca_num = min(feature_num, 250)
    if verbose:
        print('max_pca_num', max_pca_num)
    R2_list = []
    start = time.time()
    for pca_num in range(1, max_pca_num+1):
        r2 = decoding_CV(X_PCs[:, :pca_num], y)['r2'][0]
        R2_list.append(r2)
        if verbose:
            print(pca_num, 'R2', r2, 'time cost', time.time() - start)
    return R2_list

def construct_behav_regressor(behav_dt):
    predictors = []
    for episode in range(len(behav_dt['action'])):
        action = behav_dt['action'][episode] - 0.5
        stage2 = behav_dt['stage2'][episode] - 0.5
        reward = behav_dt['reward'][episode] - 0.5
        action_x_stage2 = action * stage2 * 2
        action_x_reward = action * reward * 2
        stage2_x_reward = stage2 * reward * 2

        pred_temp = np.array([action, stage2, reward, action_x_stage2, action_x_reward, stage2_x_reward]).T # (n_trials, 6)
        pred_temp_prev = np.concatenate([np.zeros([1, 6]), pred_temp[:-1]], axis=0) # (n_trials, 6)
        predictors.append(np.concatenate([pred_temp, pred_temp_prev], axis=1)) # (n_trials, 12)
    predictors = np.concatenate(predictors, axis=0)
    return predictors

def run_two_model_compare_decoding(exp_folder, neuro_data_spec):
    """
    Compre two models' predicted logits and logit changes (not successful yet)
    Maybe decode these logits from neural data (not implemented yet).
    """
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    rnn_summary = rnn_summary[(rnn_summary['rnn_type'] == 'SGRU') & (rnn_summary['outer_fold'] == 8)
                              & (rnn_summary['hidden_dim'] == 2) & (rnn_summary['readout_FC'] == True)]
    rnn_summary['model_name'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    cog_summary = cog_summary[(cog_summary['cog_type'] == 'MB0s') & (cog_summary['outer_fold'] == 8)]
    cog_summary['model_name'] = cog_summary['cog_type']
    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config),
                 neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    os.makedirs(ana_path, exist_ok=True)

    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace(
        'bin_size_', 'bs')
    for session_name in dt.behav_data_spec['session_name']:
        neuro_dt, epispode_idx, _, _ = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                         remove_nan=True, shape=2, **neuro_data_spec)
        behav_dt = dt.get_behav_data(epispode_idx, {})

        pca_results = extract_pcs(neuro_dt)
        joblib.dump(pca_results, ana_path / f'{session_name}_{fname}_pca.pkl')
        X_PCs = pca_results['PC']

        two_model_logits = {}
        for idx, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
            model_path = transform_model_format(row, source='row', target='path')
            print(model_path)
            model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
            trial_types = [model_pass['trial_type'][i] for i in epispode_idx]
            trial_types = np.concatenate(trial_types)
            model_scores = [model_pass['scores'][i] for i in epispode_idx]
            logits, logits_change = extract_value_changes(model_scores, value_type='logit')
            two_model_logits[row['model_name']] = logits_change
        from plotting_experiments.plotting import plot_2d_values
        plot_2d_values(two_model_logits['SGRU'], two_model_logits['MB0s'], trial_types,
                       x_range=(-5,5), y_range=(-5,5), x_label='SGRU', y_label='MB0s', title='', ref_line=True,
                       ref_x=0.0, ref_y=0.0, ref_diag=True, hist=False
                       )
        plt.show()
        syss

def run_decoding_exp(exp_folder, neuro_data_spec, analyses=None, ignore_analyzed=True):
    """ Run decoding experiment.

    Assume that all models in the exp sharing the same dataset.
    """
    if analyses is None:
        raise ValueError('analyses is None')
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config), neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    os.makedirs(ana_path, exist_ok=True)
    # ana_path.mkdir(exist_ok=True)
    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace('bin_size_', 'bs')
    for session_name in dt.behav_data_spec['session_name']:
        neuro_dt, epispode_idx, _, _ = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=2, **neuro_data_spec)
        behav_dt = dt.get_behav_data(epispode_idx, {})
        behav_var = construct_behav_regressor(behav_dt)
        if 'task_var_decode_neuron' in analyses:
            analysis_filename = ana_path / f'{session_name}_{fname}_task_var_decode_neuron.pkl'
            if analysis_filename.exists() and ignore_analyzed:
                print(analysis_filename, 'exists')
            else:
                neuron_decoding = decoding_CV(behav_var, neuro_dt)
                neuron_R2 = neuron_decoding['r2']
                print(session_name, 'task var predicting neuron R2>0.1 proportion', np.mean(neuron_R2>0.1))
                joblib.dump(neuron_decoding, analysis_filename)

        pca_results = extract_pcs(neuro_dt)
        joblib.dump(pca_results, ana_path / f'{session_name}_{fname}_pca.pkl')
        X_PCs = pca_results['PC']

        for idx, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
            if row['hidden_dim'] < 1:
                continue

            model_path = transform_model_format(row, source='row', target='path')
            print(model_path)
            model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
            model_scores = [model_pass['scores'][i] for i in epispode_idx]
            model_internal = [model_pass['internal'][i] for i in epispode_idx]

            ana_model_path = ANA_SAVE_PATH / model_path / 'decoding'
            os.makedirs(ana_model_path, exist_ok=True)

            _, _, full_state_vars = extract_value_changes(model_internal, value_type=0, return_full_dim=True)
            if 'value_decode_neuron' in analyses:
                analysis_filename = ana_model_path / f'{session_name}_{fname}_value.pkl'
                if analysis_filename.exists() and ignore_analyzed:
                    print(analysis_filename, 'exists')
                else:
                    neuron_decoding = decoding_CV(full_state_vars, neuro_dt)
                    neuron_R2 = neuron_decoding['r2']
                    print(session_name, model_path, 'value predicting neuron R2>0.1 proportion', np.mean(neuron_R2>0.1))
                    joblib.dump(neuron_decoding, analysis_filename)

            if 'task_var_value_decode_neuron' in analyses:
                analysis_filename = ana_model_path / f'{session_name}_{fname}_varvalue.pkl'
                if analysis_filename.exists() and ignore_analyzed:
                    print(analysis_filename, 'exists')
                else:
                    neuron_decoding = decoding_CV(np.concatenate([behav_var, full_state_vars], axis=1), neuro_dt)
                    neuron_R2 = neuron_decoding['r2']
                    print(session_name, model_path, 'task var + value predicting neuron R2>0.1 proportion', np.mean(neuron_R2>0.1))
                    joblib.dump(neuron_decoding, analysis_filename)

            # neural activity predict state variables
            y_list = []
            if 'decode_logit' in analyses:
                logits, _ = extract_value_changes(model_scores, value_type='logit')
                y_list += [('logits', logits)]
            if 'decode_logit_change' in analyses:
                _, logits_change = extract_value_changes(model_scores, value_type='logit')
                y_list += [('logits_change', logits_change)]
            if 'decode_chosen_value' in analyses and row['hidden_dim'] == 2:
                # only for cognitive models and RNNs with 2 hidden units and readout_FC=False
                chosen_values, _ = extract_value_changes(model_internal, value_type='chosen_value', action=behav_dt['action'])
                y_list += [('chosen_values', chosen_values)]
            if 'decode_value' in analyses:
                for i in range(model_internal[0].shape[1]):
                    values, _ = extract_value_changes(model_internal, value_type=i)
                    y_list += [(i, values)]

            for y_name, y in y_list:
                analysis_filename = ana_model_path / f'{session_name}_{fname}_{y_name}_R2s.pkl'
                if analysis_filename.exists() and ignore_analyzed:
                    print(analysis_filename, 'exists')
                else:
                    R2s = pca_decoding(X_PCs, y, verbose=False, max_pca_num=100)
                    joblib.dump(R2s, analysis_filename)
                    print(session_name, model_path, 'predicting', y_name, 'final R2', R2s[-1])


def compile_decoding_results(exp_folder, neuro_data_spec, extract_feature_func=None):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    rnn_summary['model_type'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    cog_summary['model_type'] = cog_summary['cog_type']

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config), neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    ana_path.mkdir(exist_ok=True)
    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace(
        'bin_size_', 'bs')
    session_names = dt.behav_data_spec['session_name']
    subset_from_kept_feat_dict = {}
    feat_scale_dict = {}

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    for session_name in session_names:
        _, _, kept_feat_idx, feat_scale = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=2, **neuro_data_spec)
        # kept_feat_idx is the shape of orginal neural data X: True means the feature is kept
        # feat_scale is the scale of all kept features; number of kept features is the same as the number of True in kept_feat_idx
        # we have customized functions to extract subset of features, in the shape of feat_scale
        if extract_feature_func is None:
            subset_from_kept_feat = np.ones_like(feat_scale, dtype=bool) # all features are kept
        else:
            subset_from_kept_feat = extract_feature_func(kept_feat_idx)
        subset_from_kept_feat_dict[session_name] = subset_from_kept_feat
        feat_scale_dict[session_name] = feat_scale

        print(session_name, 'final kept feature num', np.sum(subset_from_kept_feat))

    def compile_summary(summary, model_name_key, model_identifier_keys):
        new_rows = []
        for i, row in summary.iterrows():
            hidden_dim = row['hidden_dim']
            if hidden_dim < 1:
                continue
            model_name = row[model_name_key]
            model_path = transform_model_format(row, source='row', target='path')
            ana_model_path = ANA_SAVE_PATH / model_path / 'decoding'
            new_row = row.copy()
            new_row['trainvaltest_loss'] = (new_row['test_loss'] * new_row['test_trial_num'] +
                                            new_row['trainval_loss'] * new_row['trainval_trial_num']
                                            ) / (new_row['test_trial_num'] + new_row['trainval_trial_num'])
            total_feat_num = 0
            for session_name in session_names:
                feat_scale = feat_scale_dict[session_name]
                task_var_neuron_decoding = joblib.load(ana_path / f'{session_name}_{fname}_task_var_decode_neuron.pkl')
                task_var_mse = task_var_neuron_decoding['mse'] * feat_scale ** 2
                task_null_mse = task_var_neuron_decoding['null_mse'] * feat_scale ** 2
                task_var_value_neuron_decoding = joblib.load(
                    ana_model_path / f'{session_name}_{fname}_varvalue.pkl')
                mse = task_var_value_neuron_decoding['mse'] * feat_scale ** 2
                null_mse = task_var_value_neuron_decoding['null_mse'] * feat_scale ** 2
                assert np.isclose(null_mse, task_null_mse).all()
                subset_filter = subset_from_kept_feat_dict[session_name]
                neuron_R2 = task_var_value_neuron_decoding['r2']
                neuron_R2 = neuron_R2[subset_filter]
                # population_cpd = 1 - mse[subset_filter].sum() / task_var_mse[subset_filter].sum()
                new_row[session_name + '_sum_task_model_mse'] = mse[subset_filter].sum()
                new_row[session_name + '_sum_task_mse'] = task_var_mse[subset_filter].sum()
                new_row[session_name + '_sum_null_mse'] = null_mse[subset_filter].sum()
                new_row[session_name+'_population_cpd'] = 1 - new_row[session_name + '_sum_task_model_mse'] / new_row[session_name + '_sum_task_mse']
                new_row[session_name+'_population_R2'] = 1 - new_row[session_name + '_sum_task_model_mse'] / new_row[session_name + '_sum_null_mse']
                new_row[session_name+'_population_task_R2'] = 1 - new_row[session_name + '_sum_task_mse'] / new_row[session_name + '_sum_null_mse']
                new_row[session_name+'_mean_R2'] = np.mean(neuron_R2)
                new_row[session_name+'_R2_greater_0p1'] = np.mean(neuron_R2>0.1)
                new_row[session_name+'_pseudocell_num'] = len(neuron_R2)
                total_feat_num += len(neuron_R2)
            new_row['sum_task_mse'] = np.sum([new_row[session_name + '_sum_task_mse'] for session_name in session_names])
            new_row['sum_task_model_mse'] = np.sum([new_row[session_name + '_sum_task_model_mse'] for session_name in session_names])
            new_row['sum_null_mse'] = np.sum([new_row[session_name + '_sum_null_mse'] for session_name in session_names])
            new_row['population_cpd'] = 1 - new_row['sum_task_model_mse'] / new_row['sum_task_mse']
            new_row['population_R2'] = 1 - new_row['sum_task_model_mse'] / new_row['sum_null_mse']
            new_row['population_task_R2'] = 1 - new_row['sum_task_mse'] / new_row['sum_null_mse']
            new_row['pseudocell_num'] = total_feat_num
            new_row['mean_R2'] = np.sum([new_row[session_name+'_mean_R2']*new_row[session_name+'_pseudocell_num']
                                         for session_name in session_names]) / total_feat_num
            new_row['R2_greater_0p1'] = np.sum([new_row[session_name+'_R2_greater_0p1']*new_row[session_name+'_pseudocell_num']
                                        for session_name in session_names]) / total_feat_num
            new_rows.append(new_row)
        new_summary = pd.DataFrame(new_rows)
        agg_dict = {
            'population_cpd': ('population_cpd', 'mean'),
            'population_R2': ('population_R2', 'mean'),
            'population_task_R2': ('population_task_R2', 'mean'),
            'pseudocell_num': ('pseudocell_num', 'mean'),
            'mean_R2': ('mean_R2', 'mean'),
            'mean_R2_max': ('mean_R2', 'max'),
            'R2_greater_0p1': ('R2_greater_0p1', 'mean'),
        }
        for session_name in session_names:
            agg_dict[session_name+'_population_cpd'] = (session_name+'_population_cpd', 'mean')
            agg_dict[session_name+'_population_R2'] = (session_name+'_population_R2', 'mean')
            agg_dict[session_name+'_pseudocell_num'] = (session_name+'_pseudocell_num', 'mean')
            agg_dict[session_name+'_mean_R2'] = (session_name+'_mean_R2', 'mean')
            agg_dict[session_name+'_R2_greater_0p1'] = (session_name+'_R2_greater_0p1', 'mean')
        perf = new_summary.groupby(model_identifier_keys, as_index=False).agg(**agg_dict)
        return new_summary, perf

    rnn_summary, rnn_perf = compile_summary(rnn_summary, 'rnn_type', ['rnn_type', 'hidden_dim', 'readout_FC'])
    cog_summary, cog_perf = compile_summary(cog_summary, 'cog_type', ['cog_type', 'hidden_dim'])
    with pd_full_print_context():
        print(rnn_perf)
        print(cog_perf)
    joblib.dump(rnn_perf, ana_exp_path / f'rnn_neuron_decoding_perf_based_on_test.pkl')
    joblib.dump(rnn_summary, ana_exp_path / f'rnn_neuron_decoding_best_summary_based_on_test.pkl')
    joblib.dump(cog_perf, ana_exp_path / f'cog_neuron_decoding_perf_based_on_test.pkl')
    joblib.dump(cog_summary, ana_exp_path / f'cog_neuron_decoding_best_summary_based_on_test.pkl')

    rnn_summary.to_csv(ana_exp_path / f'rnn_neuron_decoding_best_summary_based_on_test.csv', index=False)