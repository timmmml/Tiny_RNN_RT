import os

import matplotlib.pyplot as plt
import numpy as np
import pysr
pysr.julia_helpers.init_julia()
from pysr import PySRRegressor
from analyzing_experiments.analyzing import *
from sklearn.linear_model import LinearRegression
from statsmodels.stats.proportion import proportion_confint

def construct_behav_data_spec(config):
    behav_data_spec = config['behav_data_spec']
    if isinstance(behav_data_spec, list): # assemble the dict for behavior data specification
        behav_data_spec = {k: config[k] for k in behav_data_spec}
    return behav_data_spec

def run_agent_2d_inits(ag, data_1step, lb, ub, grid_num):
    internal_all = []
    trial_type_all = []
    for x in np.linspace(lb[0], ub[0], grid_num):
        for y in np.linspace(lb[1], ub[1], grid_num):
            h0 = np.array([x,y])
            model_pass = ag.forward(data_1step, standard_output=True, h0=h0)  # a dict of lists of episodes
            model_internal = model_pass['internal']
            if isinstance(model_internal, dict): # for cog model
                model_internal = model_internal['state_var']

            internal_all.extend(model_internal)
            trial_type_all.extend(data_1step['trial_type'])
    return internal_all, trial_type_all


def run_2d_inits_exp(exp_folder, grid_num=50):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary.pkl')

    behav_1step_dt = None
    for i, row in pd.concat([cog_summary,
                             rnn_summary], axis=0, join='outer').iterrows():
        if row['hidden_dim'] != 2:
            continue
        model_path = transform_model_format(row, source='row', target='path')
        if os.path.exists(ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl'):
            print(f'2d_inits_pass for {model_path} already exists; skip')
            continue
        config = transform_model_format(row, source='row', target='config')
        ag = transform_model_format(config, source='config', target='agent')
        print(model_path)
        model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
        hid_state_shape = model_pass['hid_state_lb'].shape
        lb = np.min(model_pass['hid_state_lb']) * np.ones(hid_state_shape)
        ub = np.max(model_pass['hid_state_ub']) * np.ones(hid_state_shape)
        if behav_1step_dt is None:
            dataset = config['dataset']
            if dataset == 'SimAgent':
                if 'agent_path' in config:
                    lookup_path = config['agent_path']
                else:
                    lookup_path = config['behav_data_spec']['agent_path']
                if isinstance(lookup_path, list):
                    lookup_path = '.'.join(lookup_path)
                lookup_path = lookup_path.lower()
                if 'millerrat' in lookup_path:
                    dataset = 'MillerRat'
                elif 'bartolomonkey' in lookup_path:
                    dataset = 'BartoloMonkey'
                elif 'akamrat' in lookup_path:
                    dataset = 'AkamRat'
                else:
                    raise ValueError('dataset not recognized')
            behav_1step_dt = Dataset(dataset, behav_data_spec={'all_trial_type': True})
        if behav_1step_dt.behav_format != config['behav_format']:
            behav_1step_dt = behav_1step_dt.behav_to(config)
        data_1step = behav_1step_dt.get_behav_data(np.arange(behav_1step_dt.batch_size), config)
        internal_all, trial_type_all = run_agent_2d_inits(ag, data_1step, lb, ub, grid_num)

        joblib.dump({
            'internal': internal_all,
            'trial_type': trial_type_all,
        }, ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl')

def extract_model_par(exp_folder):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
    rnn_summary = rnn_summary[rnn_summary['readout_FC']]
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary.pkl')
    for model_type, summary in zip(['rnn_type', 'cog_type'], [rnn_summary, cog_summary]):
        for i, row in summary.iterrows():
            config = transform_model_format(row, source='row', target='config')
            ag = transform_model_format(config, source='config', target='agent')
            summary.loc[i, 'num_params'] = ag.num_params
        summary = summary.groupby([model_type, 'hidden_dim'], as_index=False).agg({'num_params':'mean'})
        print(summary)
        joblib.dump(summary, ana_exp_path / f'{model_type}_num_params.pkl')

def extract_sv_for_exp(exp_folder, verbose=False):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
    summary = summary[(summary['rnn_type'] == 'PNR1') & (summary['hidden_dim'] == 2)]

    for i, row in summary.iterrows():
        model_path = transform_model_format(row, source='row', target='path')
        ag = transform_model_format(row, source='row', target='agent')
        net = ag.model
        # T = net.lin_coef
        w = net.rnn.rnncell.weight.cpu().detach().numpy()  # H*F*I
        b = net.rnn.rnncell.bias.cpu().detach().numpy()  # H*I
        print(model_path)
        num_I = w.shape[-1]
        ev = []
        for i in range(num_I):
            u, v = np.linalg.eig(np.eye(w.shape[0]) + w[..., i])
            if verbose:
                if num_I == 4:
                    print(f'==Input {i}:', ['S1,R=0', 'S1,R=1', 'S2,R=0', 'S2,R=1', ][i])
                elif num_I == 8:
                    print(f'==Input {i}:',
                          ['A1,S1,R=0', 'A1,S1,R=1', 'A1,S2,R=0', 'A1,S2,R=1', 'A2,S1,R=0', 'A2,S1,R=1', 'A2,S2,R=0',
                           'A2,S2,R=1', ][i])
                print('W(H, F)=', w[..., i])
                print('b(H)=', b[..., i])
                print('I+W = ', np.eye(w.shape[0]) + w[..., i])
                # print('Fixed point=', -np.linalg.inv(w[...,i])@b[...,i])
                print('Eigen values & vectors of I+W:', u, v)
            ev.append({'eigenvalue': u, 'eigenvector': v, 'bias': b[..., i]})
        joblib.dump(ev, ANA_SAVE_PATH / model_path / f'eigen.pkl')

def run_scores_exp(exp_folder,best_for_test=False, model_filter=None,overwrite_config=None):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    if best_for_test:
        rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
        cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    else:
        rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
        cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary.pkl')
    if overwrite_config is None:
        overwrite_config = {}
    if model_filter is not None:
        for k,v in model_filter.items():
            if k in rnn_summary.columns:
                rnn_summary = rnn_summary[rnn_summary[k]==v]
                print(f'filter rnn_summary by {k}={v}')
            if k in cog_summary.columns:
                cog_summary = cog_summary[cog_summary[k]==v]
                print(f'filter cog_summary by {k}={v}')
            if k not in rnn_summary.columns and k not in cog_summary.columns:
                raise ValueError(f'{k} not in rnn_summary or cog_summary')
    behav_dt = None
    for i, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
        model_path = transform_model_format(row, source='row', target='path')
        if os.path.exists(ANA_SAVE_PATH / model_path / f'total_scores.pkl'):
            print(f'total_scores for {model_path} already exists; skip')
            continue
        config = transform_model_format(row, source='row', target='config')
        config.update(overwrite_config)
        ag = transform_model_format(config, source='config', target='agent')
        print(model_path)
        if behav_dt is None:
            behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config))
        if behav_dt.behav_format != config['behav_format']:
            behav_dt = behav_dt.behav_to(config)
        if 'include_embedding' in config and behav_dt.include_embedding != config['include_embedding']:
            behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config)).behav_to(config)
        data = behav_dt.get_behav_data(np.arange(behav_dt.batch_size), config)
        model_pass = ag.forward(data, standard_output=True) # a dict of lists of episodes
        model_scores = model_pass['output']
        model_internal = model_pass['internal']

        if isinstance(model_internal, dict): # for cog model
            if 'state_var' in model_internal:
                model_internal = model_internal['state_var']
            else:
                model_internal = [] # for cog model with no internal state

        if len(model_internal) > 0 and len(model_internal[0]) > 0:
            hid_state = np.concatenate(model_internal, axis=0)
            hid_state_lb = np.min(hid_state, axis=0)
            hid_state_ub = np.max(hid_state, axis=0)
        else:
            hid_state_lb = np.zeros(0)
            hid_state_ub = np.zeros(0)

        os.makedirs(ANA_SAVE_PATH / model_path, exist_ok=True)
        joblib.dump({
            'scores': model_scores,
            'internal': model_internal,
            'trial_type': data['trial_type'],
            'hid_state_lb': hid_state_lb,
            'hid_state_ub': hid_state_ub,
        }, ANA_SAVE_PATH / model_path / f'total_scores.pkl')



def get_model_summary(exp_folder):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary.pkl')
    return pd.concat([rnn_summary, cog_summary], axis=0, join='outer')

def extract_value_changes(model_scores, value_type='logit', return_full_dim=False, action=None):
    """ Extract value/logits changes from model scores

    In each block, the last trial's value has no corresponding real action/observation; thrown away
    """
    xs_change = []
    xs = []

    for episode in range(len(model_scores)):
        x = model_scores[episode]
        if value_type == 'logit':
            x = x[:, 0] - x[:, 1]
            x_change = x[1:] - x[:-1]
            x = x[:-1]
        elif value_type == 'chosen_value':
            assert action is not None
            n_trial_with_action = len(action[episode])
            assert x.shape[0] == n_trial_with_action + 1
            x = x[np.arange(n_trial_with_action), action[episode]]
            x_change = []
        elif isinstance(value_type, int):
            x = x[:, value_type]
            x_change = x[1:] - x[:-1]
            x = x[:-1]
        else:
            raise ValueError('value_type not recognized')
        xs.append(x)
        xs_change.append(x_change)
    xs = np.concatenate(xs, axis=0)
    xs_change = np.concatenate(xs_change, axis=0)
    if return_full_dim:
        full_values = np.concatenate([x[:-1] for x in model_scores], axis=0)
        return xs, xs_change, full_values
    return xs, xs_change


def extract_logit_linear_feature(logits, logits_change, trial_types):
    unique_trial_types = np.unique(trial_types)
    slopes = []
    intercepts = []
    for trial_type in unique_trial_types:
        idx = trial_types == trial_type
        logits_trial_type = logits[idx]
        logits_change_trial_type = logits_change[idx]
        reg = LinearRegression().fit(logits_trial_type.reshape([-1, 1]), logits_change_trial_type)
        slope = reg.coef_[0]
        y_intercept = reg.intercept_
        x_intercept = -y_intercept / slope
        slopes.append(-slope)
        intercepts.append(x_intercept)
    return slopes, intercepts


def extract_logit_attractor(logits, logits_change, trial_types):
    unique_trial_types = np.unique(trial_types)
    attractors = []
    for trial_type in unique_trial_types:
        idx = trial_types == trial_type
        logits_trial_type = logits[idx]
        logits_change_trial_type = logits_change[idx]
        attr = np.median(logits_trial_type[np.argsort(np.abs(logits_change_trial_type))[:10]])
        attractors.append(attr)

    return attractors

def extract_logit_action_freq(scores, trial_types):
    unique_trial_types = np.unique(np.concatenate(trial_types))
    if len(unique_trial_types) == 4:
        action = [(tt//2).astype(int) for tt in trial_types]
    elif len(unique_trial_types) == 8:
        action = [(tt//4).astype(int) for tt in trial_types]
    else:
        raise ValueError('trial type not recognized')
    assert len(scores[0]) == len(trial_types[0]) + 1
    logits, logits_change = extract_value_changes(scores, value_type='logit')
    bin_size = 0.5
    bin_num = 300
    bin_results = {}
    bin_centers = np.linspace(logits.min()+bin_size/2, logits.max()-bin_size/2, bin_num, endpoint=True)
    for trial_type in unique_trial_types:
        logits_trial_type = logits[np.concatenate(trial_types) == trial_type]
        # bin_centers = np.linspace(logits_trial_type.min()+bin_size/2, logits_trial_type.max()-bin_size/2, bin_num, endpoint=True)
        # action_counts_of_bin = np.zeros([bin_num, 2]) # A1, A2
        action_counts_of_bin = np.ones([bin_num, 2])  # A1, A2 # add 1 as prior
        for bin_idx in range(bin_num):
            bin_left = bin_centers[bin_idx] - bin_size/2
            bin_right = bin_centers[bin_idx] + bin_size/2
            # bin_left = np.percentile(logits_trial_type, bin_left*100)
            # bin_right = np.percentile(logits_trial_type, bin_right*100)
            # center_of_bin[bin_idx] = np.percentile(logits_trial_type, bin_edges[bin_idx]*100)

            for epi_idx in range(len(scores)):
                epi_logits = scores[epi_idx][:-1, 0] - scores[epi_idx][:-1, 1]
                idx = np.where((trial_types[epi_idx] == trial_type) & (epi_logits >= bin_left) & (epi_logits <= bin_right))[0]
                if len(idx) == 0:
                    continue
                idx += 1 # consider the next action
                if idx[-1] == len(action[epi_idx]): # last trial has no following action
                    idx = idx[:-1]
                action_counts_of_bin[bin_idx, 0] += np.sum(action[epi_idx][idx] == 0)
                action_counts_of_bin[bin_idx, 1] += np.sum(action[epi_idx][idx] == 1)
        p = action_counts_of_bin[:, 0] / np.sum(action_counts_of_bin, axis=1)
        ci_low = np.zeros(bin_num)
        ci_upp = np.zeros(bin_num)
        for bin_idx in range(bin_num):
            ci_low[bin_idx], ci_upp[bin_idx] = proportion_confint(
                action_counts_of_bin[bin_idx, 0], action_counts_of_bin[bin_idx].sum(), alpha=0.32, method='beta')
        # plt.plot(bin_centers, p, color='C'+str(trial_type))
        bin_results[trial_type] = (bin_centers, p, ci_low, ci_upp, action_counts_of_bin)
    return bin_results

def action_freq_after_action_seq(exp_folder):
    model_summary = get_model_summary(exp_folder)
    model_summary = model_summary[model_summary['hidden_dim'] == 1]
    config = model_summary.iloc[0]['config']
    behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config))
    batch_size = behav_dt.batch_size
    behav_dt = behav_dt.get_behav_data(list(range(batch_size)))
    action0_reward1_cases = {}
    action1_reward1_cases = {}
    for epi_idx in range(batch_size):
        action0_reward1_counter = 0
        action1_reward1_counter = 0
        act = behav_dt['action'][epi_idx]
        rew = behav_dt['reward'][epi_idx]
        for trial in range(len(act)):
            if act[trial] != 0 or rew[trial] != 1:
                # before this trial, action0_reward1_counter is the number of action0_reward1 cases
                action0_reward1_cases.setdefault(action0_reward1_counter, []).append(act[trial])
                action0_reward1_counter = 0
            if act[trial] != 1 or rew[trial] != 1:
                action1_reward1_cases.setdefault(action1_reward1_counter, []).append(act[trial])
                action1_reward1_counter = 0
            if act[trial] == 0 and rew[trial] == 1:
                action0_reward1_counter += 1
            if act[trial] == 1 and rew[trial] == 1:
                action1_reward1_counter += 1
    # sort keys
    action0_reward1_cases = {k: action0_reward1_cases[k] for k in sorted(action0_reward1_cases.keys())}
    action1_reward1_cases = {k: action1_reward1_cases[k] for k in sorted(action1_reward1_cases.keys())}
    print('action0_reward1_cases', {k: 1-np.mean(action0_reward1_cases[k]) for k in action0_reward1_cases.keys()})
    print('action1_reward1_cases', {k: np.mean(action1_reward1_cases[k]) for k in action1_reward1_cases.keys()})
    plt.figure()
    for k, v in action0_reward1_cases.items():
        v = np.array(v)
        pr = np.mean(v == 0)
        plt.scatter(k, pr, color='C1')
        ci_low, ci_upp = proportion_confint((v==0).sum(), len(v), alpha=0.32, method='beta')
        plt.plot([k, k], [ci_low, ci_upp], color='C1')
    for k, v in action1_reward1_cases.items():
        v = np.array(v)
        pr = np.mean(v == 1)
        plt.scatter(k, pr, color='C3')
        ci_low, ci_upp = proportion_confint((v == 1).sum(), len(v), alpha=0.32, method='beta')
        plt.plot([k, k], [ci_low, ci_upp], color='C3')
    plt.show()


def logit_vs_action_freq(exp_folder):
    model_summary = get_model_summary(exp_folder)
    model_summary = model_summary[model_summary['hidden_dim'] == 1]

    for i, row in model_summary.iterrows():
        config = row['config']
        if 'rnn_type' in config and config['rnn_type'] == 'SGRU':
        # if 'cog_type' in config and config['cog_type'] == 'MB0s':
        # if 'cog_type' in config and config['cog_type'] == 'LS0':
        # if 'cog_type' in config and config['cog_type'] == 'LS1':
            model_path = row['model_path']
            model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
            scores = model_pass['scores']
            trial_types = model_pass['trial_type']
            plt.figure()
            extract_logit_action_freq(scores, trial_types)
            plt.xlabel('logit')
            plt.ylabel('P(A1)')
            plt.xlim([-5, 5])
            plt.ylim([0, 1])
            plt.show()
            syss
            break




    #     table = table[[0,2],:].astype(int)
    #     odds_ratio, p_value = fisher_exact(table)
    #     plt.subplot(3, 1, 2) # change y axis is log
    #     ax = plt.gca()
    #     ax.set_yscale('log')
    #     plt.scatter(bin_center, odds_ratio, color='r')
    #     plt.scatter(bin_center, 1, color='k')
    #     plt.ylabel('odds_ratio')
    #     plt.yticks([0.1, 1, 10])
    #     plt.subplot(3, 1, 3)
    #     ax.set_yscale('log')
    #     plt.scatter(bin_center, p_value, color='r')
    #     plt.ylabel('p_value')
    #     plt.yticks([0.001, 0.01, 0.1, 1])
    #         # ratios.append(p)
    #         # ratios_std.append(np.sqrt(p*(1-p)/all_action_count))
    #         # print(f'bin {bin_idx}, left{left}, right{right} trial type {trial_type}, all count {all_action_count}, A1 count {A1_action_count}, ratio {A1_action_count/all_action_count}')
    # # plt.xlabel('logit')
    # plt.xlabel('P(A1)')
    # plt.show()

def extract_1d_logit_for_exp(exp_folder):
    model_summary = get_model_summary(exp_folder)
    model_summary = model_summary[model_summary['hidden_dim'] == 1]
    slope_summary = []
    intercept_summary = []
    trial_type_str = []
    for i, row in model_summary.iterrows():
        config = row['config']
        model_type = config['rnn_type'] if 'rnn_type' in config else config['cog_type']
        model_path = row['model_path']
        model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
        model_scores = model_pass['scores']
        trial_types = model_pass['trial_type']
        trial_types = np.concatenate(trial_types)
        logits, logits_change = extract_value_changes(model_scores)
        print(model_path)
        # print(logits, logits_change)
        num_trial_type = len(np.unique(trial_types))
        if len(trial_type_str) == 0:
            if num_trial_type == 4:
                trial_type_str = ['S0R0', 'S0R1', 'S1R0', 'S1R1']
            elif num_trial_type == 8:
                trial_type_str = ['A0S0R0', 'A0S0R1', 'A0S1R0', 'A0S1R1', 'A1S0R0', 'A1S0R1', 'A1S1R0', 'A1S1R1']
            else:
                raise ValueError
        slope_row = row.copy()
        intercept_row = row.copy()
        slope_row['model_type'] = model_type
        intercept_row['model_type'] = model_type
        slopes, intercepts = extract_logit_linear_feature(logits, logits_change, trial_types)
        if model_type not in ['PNR1', 'MB0s', 'MFs']:
            intercepts = extract_logit_attractor(logits, logits_change, trial_types)
        for i, s in enumerate(slopes):
            slope_row[trial_type_str[i]] = s
        slope_summary.append(slope_row)
        for i, s in enumerate(intercepts):
            intercept_row[trial_type_str[i]] = s
        intercept_summary.append(intercept_row)
    slope_summary = pd.DataFrame(slope_summary)
    intercept_summary = pd.DataFrame(intercept_summary)
    slope_group_summary = pd.concat([slope_summary.groupby('model_type')[t].apply(list) for t in trial_type_str], axis=1, join='inner').reset_index()
    intercept_group_summary = pd.concat([intercept_summary.groupby('model_type')[t].apply(list) for t in trial_type_str], axis=1, join='inner').reset_index()
    for t in trial_type_str:
        slope_group_summary[t+'_mean'] = slope_group_summary[t].apply(np.mean)
        intercept_group_summary[t+'_mean'] = intercept_group_summary[t].apply(np.mean)
    for t in trial_type_str:
        slope_group_summary[t+'_std'] = slope_group_summary[t].apply(np.std)
        intercept_group_summary[t+'_std'] = intercept_group_summary[t].apply(np.std)
    reindex = ['model_type'] + [t+'_mean' for t in trial_type_str] + [t+'_std' for t in trial_type_str]+ [t for t in trial_type_str]
    slope_group_summary = slope_group_summary.reindex(reindex, axis=1)
    intercept_group_summary = intercept_group_summary.reindex(reindex, axis=1)
    with pd_full_print_context():
        print(slope_group_summary)
        print(intercept_group_summary)

    ana_exp_path = ANA_SAVE_PATH / exp_folder
    joblib.dump(slope_summary, ana_exp_path / 'slope_summary.pkl')
    joblib.dump(intercept_summary, ana_exp_path / 'intercept_summary.pkl')
    joblib.dump(slope_group_summary, ana_exp_path / 'slope_group_summary.pkl')
    joblib.dump(intercept_group_summary, ana_exp_path / 'intercept_group_summary.pkl')


def compute_nts_pattern_features(utility_pattern):
    assert len(utility_pattern) == 8
    A1S1R0, A1S1R1, A1S2R0, A1S2R1, A2S1R0, A2S1R1, A2S2R0, A2S2R1 = utility_pattern
    feat = {}
    # usually A1 has utility_pattern > 0, A2 has utility_pattern < 0
    feat['action_asymmetry'] = (np.abs(A1S1R0 + A2S1R0) + np.abs(A1S1R1 + A2S1R1) + np.abs(A1S2R0 + A2S2R0) + np.abs(A1S2R1 + A2S2R1)) / 4
    feat['state_dependence'] = (np.abs(A1S1R0 - A1S2R0) + np.abs(A1S1R1 - A1S2R1) + np.abs(A2S1R0 - A2S2R0) + np.abs(A2S1R1 - A2S2R1)) / 4
    feat['reward_dependence'] = (np.abs(A1S1R0 - A1S1R1) + np.abs(A1S2R0 - A1S2R1) + np.abs(A2S1R0 - A2S1R1) + np.abs(A2S2R0 - A2S2R1))/4
    # feat['one_reward_action_asymmetry'] = (np.abs(A1S1R1 + A2S1R1) + np.abs(A1S2R1 + A2S2R1)) / 2
    # feat['one_reward_state_dependence'] = (np.abs(A1S1R1 - A1S2R1) + np.abs(A2S1R1 - A2S2R1)) / 2
    feat['one_reward_utility'] = (A1S1R1 + A1S2R1 - (A2S1R1 + A2S2R1))/4
    # feat['one_reward_utility_std'] = np.ptp([A1S1R1, A1S2R1, -A2S1R1, -A2S2R1])
    feat['no_reward_utility'] = (A1S1R0 + A1S2R0 - (A2S1R0 + A2S2R0))/4
    # feat['no_reward_utility_std'] = np.ptp([A1S1R0, A1S2R0, -A2S1R0, -A2S2R0])
    return feat

def compute_nts_pattern_slope_features(slope_group_summary):
    feat = {}
    feat['MF_slope'] = slope_group_summary[slope_group_summary['model_type'] == 'MFs']['A0S0R0_mean'].iloc[0]
    slope_group_summary = slope_group_summary[slope_group_summary['model_type'] == 'SGRU']
    for col in slope_group_summary.columns:
        if 'mean' in col:
            feat[col+'_slope'] = slope_group_summary[col].iloc[0]
    feat['one_reward_slope'] = (feat['A0S0R1_mean_slope'] + feat['A0S1R1_mean_slope'] + feat['A1S0R1_mean_slope'] + feat['A1S1R1_mean_slope']) / 4
    feat['no_reward_slope'] = (feat['A0S0R0_mean_slope'] + feat['A0S1R0_mean_slope'] + feat['A1S0R0_mean_slope'] + feat['A1S1R0_mean_slope']) / 4
    feat['slope_diff'] = feat['one_reward_slope'] - feat['no_reward_slope']
    feat['slope_ratio'] = feat['one_reward_slope'] / feat['no_reward_slope']
    feat['avg_slope'] = (feat['one_reward_slope'] + feat['no_reward_slope']) / 2
    return feat

def compile_1d_logit_for_exps(exp_folders, compile_exp_folder):
    """Compile logit for all exps in exp_folders (Akam rat data)
    """
    subject_summary = joblib.load(ANA_SAVE_PATH / 'AkamRat' / 'subject_summary.pkl')

    cond_patterns_all = []
    for exp_folder in exp_folders:
        ana_exp_path = ANA_SAVE_PATH / exp_folder
        intercept_group_summary = joblib.load(ana_exp_path / 'intercept_group_summary.pkl')
        intercept_group_summary = intercept_group_summary[intercept_group_summary['model_type'] == 'SGRU']
        cond_pattern = []
        for col in intercept_group_summary.columns:
            if 'mean' in col:
                cond_pattern.append(intercept_group_summary[col].iloc[0])
        cond_pattern = np.array(cond_pattern)
        cond_pattern /= np.abs(cond_pattern).max()
        cond_patterns_all.append(cond_pattern)
    # add the MF pattern
    # cond_patterns_all.append([0, 1, 0, 1, 0, -1, 0, -1]) # MF
    # exp_folders.append('exp_seg_akamratMF')

    cond_patterns_all = np.array(cond_patterns_all) # num_exp x num_cond
    print(cond_patterns_all)
    print(cond_patterns_all.shape)
    # _dimension_reduction_for_1d_logit(cond_patterns_all, exp_folders, compile_exp_folder)

    feat_dt = []
    for i, exp_folder in enumerate(exp_folders):
        cond_pattern = cond_patterns_all[i]
        ID = int(exp_folder[15:]) # remove "exp_seg_akamrat"
        ID_summary = subject_summary[subject_summary['subject_ID'] == ID]
        feat = {'animal': ID, 'block_length': ID_summary['non_neutral_block_length'].iloc[0]}
        feat |= compute_nts_pattern_features(cond_pattern)

        ana_exp_path = ANA_SAVE_PATH / exp_folder
        slope_group_summary = joblib.load(ana_exp_path / 'slope_group_summary.pkl')
        feat |= compute_nts_pattern_slope_features(slope_group_summary)

        feat_dt.append(feat)

    feat_dt = pd.DataFrame(feat_dt)
    with pd_full_print_context():
        print(feat_dt)

    joblib.dump(feat_dt, ANA_SAVE_PATH / compile_exp_folder / 'logit_1d_pattern_features.pkl')
    # _statsmodel_pred(feat_dt)
    # _seaborn_pairwise(feat_dt, compile_exp_folder)


def _seaborn_pairwise(feat_dt, compile_exp_folder):
    # use seaborn to make pairwise comparison for each pair of two features by scatter plot
    import seaborn as sns
    from scipy.stats import pearsonr

    def corrfunc(x, y, **kws):
        (r, p) = pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f} ".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
        ax.annotate("p = {:.3f}".format(p),
                    xy=(.4, .9), xycoords=ax.transAxes)


    g = sns.pairplot(feat_dt,kind='reg')
    # for s1 in range(g.axes.shape[0]):
    #     for s2 in range(g.axes.shape[1]):
    #         if s1 == s2:
    #             continue
    #         g.axes[s1, s2].set_xlim((-0.1, 1.1))
    #         g.axes[s1, s2].set_xticks([0, 1])
    #         g.axes[s1, s2].set_ylim((-0.1, 1.1))
    #         g.axes[s1, s2].set_yticks([0, 1])

    g.map(corrfunc)
    plt.savefig(FIG_PATH / compile_exp_folder / 'pairwise_comparison.png')
    plt.close()

def _dimension_reduction_for_1d_logit(cond_patterns_all, exp_folders, compile_exp_folder):
    """dimension reduction using tsne and then make a plot
    also use pca
    results not successful
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from adjustText import adjust_text
    for seed in range(10):
        tsne = TSNE(n_components=2, random_state=seed, perplexity=5)
        pca = PCA(n_components=2, random_state=seed)
        for method, model in zip([#'tsne',
                                  'pca'],
                                 [#tsne,
                                  pca]):
            cond_patterns_all_2d = model.fit_transform(cond_patterns_all)
            plt.figure()
            plt.scatter(cond_patterns_all_2d[:, 0], cond_patterns_all_2d[:, 1])
            texts = []
            for i, exp_folder in enumerate(exp_folders):
                texts.append(plt.text(cond_patterns_all_2d[i, 0], cond_patterns_all_2d[i, 1], exp_folder[15:]))
            adjust_text(texts)
            os.makedirs(FIG_PATH / compile_exp_folder, exist_ok=True)
            plt.savefig(FIG_PATH / compile_exp_folder / f'1d_attractor_pattern_{method}_seed{seed}.png')
            plt.show()

def _statsmodel_pred(feat_dt):
    # using statsmodels and stepwise regression
    # predict block_length using: action_asymmetry  state_dependence  reward_dependence  one_reward_utility  no_reward_utility
    #
    import statsmodels.api as sm
    X = feat_dt[['action_asymmetry', 'state_dependence', 'reward_dependence', 'one_reward_utility','no_reward_utility']]
    y = feat_dt['block_length']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)

    results = model.fit()
    print(results.summary())

    model = forward_selected(pd.concat([y, X], axis=1), 'block_length')
    print(model.summary())

def compare_logit_rt(exp_folder):
    """Try to compare the logit change with the RT change but not very successful"""
    model_summary = get_model_summary(exp_folder)
    model_summary = model_summary[model_summary['hidden_dim'] == 1]
    config = model_summary.iloc[0]['config']
    behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config))
    behav_dt = behav_dt.get_behav_data(list(range(behav_dt.batch_size)))
    reaction_time = behav_dt['reaction_time']
    target_time = behav_dt['target_time']
    reward_time = behav_dt['reward_time']
    for i, row in model_summary.iterrows():
        config = row['config']
        model_type = config['rnn_type'] if 'rnn_type' in config else config['cog_type']
        if model_type != 'SGRU':
            continue
        model_path = row['model_path']
        model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
        model_scores = model_pass['scores']
        trial_types = model_pass['trial_type']
        trial_types = np.concatenate(trial_types)
        logits, logits_change = extract_value_changes(model_scores)
        logits_change = logits_change.reshape(96,60)[:,:-1].reshape(-1)
        trial_types = trial_types.reshape(96,60)[:,:-1].reshape(-1)
        pre_logits = logits.reshape(96,60)[:,:-1].reshape(-1)
        logits = logits.reshape(96,60)[:,1:].reshape(-1)
        cared_time_align_logit_change = np.concatenate([rt[1:] for rt in reaction_time])
        # correlation between logit at trial t and RT at trial t
        # corrlations between logit change/trial type at trial t-1 and RT at trial t
        colors = np.array(['C0','C1','C2','C3',])[trial_types]
        plt.subplot(3, 2, 1)
        plt.scatter(logits, cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('logit')
        plt.ylabel('reaction time')
        plt.subplot(3, 2, 2)
        plt.scatter(np.abs(logits), cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('|logit|')
        plt.ylabel('reaction time')
        plt.subplot(3, 2, 3)
        plt.scatter(logits_change, cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('logit change')
        plt.ylabel('reaction time')
        plt.subplot(3, 2, 4)
        plt.scatter(np.abs(logits_change), cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('|logit change|')
        plt.ylabel('reaction time')

        plt.subplot(3, 2, 5)
        plt.scatter(pre_logits, cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('logit at trial t-1')
        plt.ylabel('reaction time')
        plt.subplot(3, 2, 6)
        plt.scatter(np.abs(pre_logits), cared_time_align_logit_change, color=colors, alpha=0.3)
        plt.title(f'{model_type}')
        plt.xlabel('|logit at trial t-1|')
        plt.ylabel('reaction time')
        plt.show()

        import seaborn as sns
        plt.figure()
        # plot distribution of pre_logits with sns for different trial types
        plt.subplot(1, 2, 1)
        for tt in np.unique(trial_types):
            sns.distplot(pre_logits[trial_types==tt], color=colors[trial_types==tt][0])
        plt.title(f'{model_type}')
        plt.xlabel('logit at trial t-1')
        plt.ylabel('density')
        plt.subplot(1, 2, 2)
        for tt in np.unique(trial_types):
            sns.distplot(logits[trial_types == tt], color=colors[trial_types == tt][0])
        plt.title(f'{model_type}')
        plt.xlabel('logit at trial t')
        plt.ylabel('density')
        plt.show()

        # calculate correlation and p value
        # from scipy.stats import pearsonr
        # for tt in np.unique(trial_types):
        #     print(f'logit change vs reaction time, trial type {tt}')
        #     print(pearsonr(np.abs(logits_change[trial_types==tt]), cared_time_align_logit_change[trial_types==tt]))


        syss

def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    import statsmodels.formula.api as smf
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


def sym_regression_for_exp(exp_folder):
    model_summary = get_model_summary(exp_folder)
    for i, row in model_summary.iterrows():
        hidden_dim = row['hidden_dim']
        if hidden_dim < 1 or hidden_dim > 2:
            continue
        model_path = row['model_path']
        sym_regression_for_model(model_path)

def sym_regression_for_model(model_path):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_output = model_pass['internal']
    hidden_dim = model_output[0].shape[1]
    trial_types = model_pass['trial_type']
    trial_types = np.concatenate(trial_types)
    unique_trial_type_num = len(np.unique(trial_types))
    symreg_X = []
    symreg_y = []
    symreg_note = []
    for d in range(hidden_dim):
        _, values_change, full_values = extract_value_changes(model_output, value_type=d, return_full_dim=True)
        for trial_type in range(unique_trial_type_num):
            idx = trial_types == trial_type
            symreg_X.append(full_values[idx])
            symreg_y.append(values_change[idx])
            symreg_note.append(f'hidden_{d}_trialtype_{trial_type}')
    os.makedirs(ANA_SAVE_PATH / model_path / 'symreg', exist_ok=True)
    for i in range(len(symreg_X)):
        sym_regression_for_condition(symreg_X[i], symreg_y[i], symreg_note[i], ANA_SAVE_PATH / model_path / 'symreg')


def sym_regression_for_condition(X, y, note, save_path):
    model = PySRRegressor(
        procs=16,
        populations=32,
        progress=False,
        update=False,
        warm_start=False,
        model_selection="best",  # Result is mix of simplicity+accuracy
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "inv(x) = 1/x",
            # "cos","sin",
            # "square", "cube",
            # "exp",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        loss="loss(x, y) = (x - y)^2",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(X, y)
    with open(save_path / f'{note}.txt', 'w') as f:
        with pd_full_print_context():
            print(model, file=f)
