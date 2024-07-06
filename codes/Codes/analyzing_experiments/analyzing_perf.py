import pandas as pd
from utils import set_os_path_auto
from .analyzing import *
from .analyzing_check import check_missing_models

def combine_exp_summary(exp_folder, id_keys=None, filter_dict=None):
    """Combine the summary of all models in the experiment.

    Args:
        exp_folder (str): The name of the experiment folder.
        id_keys (list): The list of keys to be added to the summary, should uniquely identify the model.

    Returns:
        dataframe: The combined summary.
    """
    if id_keys is None:
        raise ValueError('id_keys should be specified.')
    if filter_dict is None:
        filter_dict = {}
    combined_summary = pd.DataFrame()
    path = MODEL_SAVE_PATH / Path(exp_folder)
    summary_paths = []
    for p in path.rglob("*"): # recursively search all subfolders
        if p.name == 'allfold_summary.pkl':
            summary_paths.append(p)

    for summary_path in summary_paths:
        with set_os_path_auto():
            summary = joblib.load(summary_path)
        filter_flag = True
        for k in filter_dict:
            # if the dataframe (and the config column) does not have the key or value, then pass
            if (k not in summary.columns or summary[k].iloc[0] != filter_dict[k]) and \
                ( k not in summary.iloc[0].config.keys() or summary.iloc[0].config[k] != filter_dict[k]):
                filter_flag = False
                break
        if not filter_flag:
            continue
        summary['exp_model_path'] = summary_path.parent.name
        combined_summary = pd.concat([combined_summary, summary], axis=0, join='outer')
    combined_summary = combined_summary.reset_index(drop=True)
    if len(combined_summary) == 0:
        raise ValueError('No summary found!')
    for key in id_keys:
        # insert a new column called key; for each row, the value is read from config dict
        if key not in combined_summary.columns:
            combined_summary[key] = combined_summary.apply(lambda row: 'none' if key not in row['config'] else row['config'][key],
                                                           axis=1)

    sub_combined_summary = combined_summary[id_keys].drop_duplicates(inplace=False)
    if len(sub_combined_summary) != len(combined_summary):
        with pd_full_print_context():
            print(id_keys)
            print(len(sub_combined_summary))
            print(len(combined_summary))
        raise ValueError('Warning: the id_keys do not uniquely identify the model.')
    return combined_summary


def select_best_models_by_keys(df, group_by_keys=None, perf_key='', select_func='min'):
    """Select the best models based on the performance.
    We first group the models by group_by_keys, and then select the best model in each group.

    Args:
        df (dataframe): The dataframe of the summary.
        group_by_keys (list, optional): The list of keys to group the models. Defaults to None.
        perf_key (str, optional): The key of the performance. Defaults to ''.
        select_func (str, optional): The function to select the best model. Defaults to 'min'.

    Returns:
        dataframe: The dataframe of the best models.
    """
    if group_by_keys is None:
        raise ValueError('group_by_keys should not be None')
    if perf_key == '':
        raise ValueError('perf_key should not be empty')
    if select_func == 'min':
        select_func_ = np.argmin
    elif select_func == 'max':
        select_func_ = np.argmax
    else:
        raise ValueError('select_func not recognized')

    best_df = pd.DataFrame()
    for name, group in df.groupby(group_by_keys):
        best_idx = select_func_(group[perf_key])
        best_df = pd.concat([best_df, group.iloc[best_idx:best_idx+1]], axis=0, join='outer')
    best_df = best_df.reset_index(drop=True)
    return best_df
    # df_new = df[df.groupby(group_by_keys)[perf_key].transform(select_func) == df[perf_key]]
    # df_new = df_new.reset_index(drop=True)
    # return df_new


def select_final_agent_perf(exp_folder, model_identifier_keys=None, cv_keys=None, compete_from_keys=None, filter_dict=None, inner_fold_perf_key='trainval_loss', filter_dict_for_summary=None):
    # filter_dict only consider 'rnn_type' & 'cog_type', used before the summary is generated
    # select all models satisfying the filter_dict
    summary = combine_exp_summary(exp_folder, model_identifier_keys+cv_keys+compete_from_keys, filter_dict=filter_dict)
    model_identifier_keys = model_identifier_keys.copy()
    if filter_dict_for_summary is not None: # filter_dict_for_summary used after the summary is generated
        for k, v in filter_dict_for_summary.items():
            if not isinstance(v, list):
                v = [v]
            summary = summary[summary[k].isin(v)]
            model_identifier_keys.remove(k) # remove the key from model_identifier_keys, because we want to combine these models later
        filter_dict_for_summary_keys = list(filter_dict_for_summary.keys()) # e.g. ['rnn_type']
    else:
        filter_dict_for_summary_keys = []
    # select the best model with the lowest validation loss on compete_from_keys for each outer and inner fold
    summary = select_best_models_by_keys(summary, group_by_keys=model_identifier_keys+filter_dict_for_summary_keys+cv_keys, perf_key='val_loss', select_func='min')

    # compute the performance SEM of the selected models on the test set, SEM over inner folds, then average over outer folds
    perf_inner_sem = summary.groupby(model_identifier_keys+['outer_fold'], as_index=False).agg(test_loss_inner_sem=('test_loss', 'sem'))
    perf_mean_inner_sem = perf_inner_sem.groupby(model_identifier_keys, as_index=False).agg(test_loss_mean_inner_sem=('test_loss_inner_sem', 'mean'))

    # summary = insert_model_test_scores_in_df(summary)
    # combine_test_scores(summary, model_identifier_keys+['outer_fold'])
    # select the best model with lowest train-val (or test) loss on inner fold
    assert inner_fold_perf_key in ['trainval_loss', 'test_loss']
    summary = select_best_models_by_keys(summary, group_by_keys=model_identifier_keys+['outer_fold'], perf_key=inner_fold_perf_key, select_func='min')
    # average over outer fold to obtain the final CV test performance (stored in perf)
    summary['total_test_loss'] = summary.apply(lambda row: row['test_loss']*row['test_trial_num'], axis=1)
    summary['total_trainval_loss'] = summary.apply(lambda row: row['trainval_loss']*row['trainval_trial_num'], axis=1)
    summary['total_train_loss'] = summary.apply(lambda row: row['train_loss']*row['train_trial_num'], axis=1)
    summary['total_val_loss'] = summary.apply(lambda row: row['val_loss']*row['val_trial_num'], axis=1)
    if 'trainval_percent' in summary.columns:
        print('Warning: 9->5, 18->10')
        summary['trainval_percent'] = summary['trainval_percent'].apply(lambda x: 5 if x == 9 else (10 if x == 18 else x))
    perf = summary.groupby(model_identifier_keys, as_index=False).agg(
        agg_outer_fold=('outer_fold', list),
        agg_test_loss=('test_loss', list),
        total_test_loss=('total_test_loss','sum'),
        test_trial_num=('test_trial_num','sum'),

        #agg_trainval_loss=('trainval_loss', list),
        total_trainval_loss=('total_trainval_loss','sum'),
        trainval_trial_num=('trainval_trial_num','sum'),

        # agg_train_loss=('train_loss', list),
        total_train_loss=('total_train_loss','sum'),
        train_trial_num=('train_trial_num','sum'),

        # agg_val_loss=('val_loss', list),
        total_val_loss=('total_val_loss','sum'),
        val_trial_num=('val_trial_num','sum'),

        test_loss_outer_std=('test_loss','std'),
        test_loss_outer_sem=('test_loss','sem'),
        mean_train_trial_num=('train_trial_num','mean'),
    )
    perf['test_loss_mean_inner_sem'] = perf_mean_inner_sem['test_loss_mean_inner_sem']
    perf['test_loss'] = perf['total_test_loss']/perf['test_trial_num']
    perf['trainval_loss'] = perf['total_trainval_loss']/perf['trainval_trial_num']
    perf['train_loss'] = perf['total_train_loss']/perf['train_trial_num']
    perf['val_loss'] = perf['total_val_loss']/perf['val_trial_num']
    return perf, summary


def select_final_rnn_perf(exp_folder, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss', return_dim_est=False, combine_model_then_select=None):
    if additional_keys is None:
        additional_keys = {}
    filter_dict = {'agent_type': 'RNN'}
    model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC'] + additional_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', []) # best model from inner fold and average over outer fold
    compete_from_keys = ['l1_weight', 'seed'] + additional_keys.setdefault('compete_from_keys', []) # the keys to pick the best model instance
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict, inner_fold_perf_key=inner_fold_perf_key, filter_dict_for_summary=combine_model_then_select)
    if combine_model_then_select is not None:
        new_rnn_type = '+'.join(combine_model_then_select['rnn_type'])
        perf['rnn_type'] = new_rnn_type
        summary['rnn_type'] = new_rnn_type
    perf.insert(1, 'test_loss', perf.pop('test_loss'))
    perf.insert(2, 'trainval_loss', perf.pop('trainval_loss'))
    perf.insert(3, 'train_loss', perf.pop('train_loss'))
    perf.insert(4, 'val_loss', perf.pop('val_loss'))
    if verbose:
        with pd_full_print_context():
            print(perf)
    # estimate dimensionality
    perf_gru = perf[perf['readout_FC'] == True].reset_index(drop=True)
    # given a hidden dimension, select the best model with the lowest test loss (from GRU/SGRU/PNR1)
    perf_gru = select_best_models_by_keys(perf_gru, group_by_keys=['hidden_dim'], perf_key='test_loss', select_func='min')

    perf_gru['less_pvalue'] = [np.zeros(len(perf_gru)) for _ in range(len(perf_gru))]
    perf_gru['less_than_former'] = [1 for _ in range(len(perf_gru))]
    from scipy.stats import ttest_rel
    for i, i_row in perf_gru.iterrows():
        L = i_row['agg_outer_fold']
        assert all(L[i] < L[i+1] for i in range(len(L) - 1)), L
        for j, j_row in perf_gru.iterrows():
            pvalue = ttest_rel(i_row['agg_test_loss'], j_row['agg_test_loss'], alternative='less').pvalue
            perf_gru.loc[i, 'less_pvalue'][j] = pvalue
            if j<i and pvalue > 0.05:
                perf_gru.loc[i, 'less_than_former'] = 0
    if verbose:
        with pd_full_print_context():
            print('Estimated dimensionality:')
            if 'rnn_type' in perf_gru.columns:
                print(perf_gru[['rnn_type', 'hidden_dim', 'less_than_former', 'test_loss', 'less_pvalue']])
    if return_dim_est:
        return perf, summary, perf_gru
    return perf, summary


def select_final_cog_perf(exp_folder, agent_type, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss'):
    """
    Select the final performance of the cognitive model.

    Args:
        exp_folder (str): The folder of the experiment.
        agent_type (str): The type of the cognitive model. PRLCog, RTSCog, NTSCog...
    """
    cog_hidden_dim = {
        'BAS': 0, 'MB0s': 1, 'MFs': 1, 'LS0': 1, 'LS1': 1, 'MB0se': 1, 'MBsvlr': 1,'MBsflr': 1,
        'MB0': 2, 'MB1': 2, 'MB0md': 2, 'MB0mdnb': 2, 'MB0m': 2, 'Q(1)': 2, 'MBsah': 2,
        'Q(0)': 4, 'RC': 8,
        'MF_MB_bs_rb_ck': 6,#
        'MF_bs_rb_ck': 4,#
        'MB_bs_rb_ck': 4,#
        'MF_MB_dec_bs_rb_ck': 6,#
        'MF_dec_bs_rb_ck': 4,#
        'MB_dec_bs_rb_ck': 4,#
        'MF_MB_vdec_bs_rb_ck': 6,#
        'MF_MB_bs_rb_ec': 6 + 1, #
        'MF_MB_vdec_bs_rb_ec': 6 + 1, #
        'MF_MB_dec_bs_rb_ec': 6 + 1,#
        'MF_MB_dec_bs_rb_mc': 6 + 2,#
        'MF_MB_dec_bs_rb_ec_mc': 6 + 3,#
        'MFmoMF_MB_dec_bs_rb_ec_mc': 6 + 4 + 3, #
        'MFmoMF_dec_bs_rb_ec_mc': 4 + 4 + 3,#
        'MB_dec_bs_rb_ec_mc': 4 + 3,
        'AC3': 27,
        'AC4': 27,
    }
    if additional_keys is None:
        additional_keys = {}
    model_identifier_keys = ['cog_type'] + additional_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', []) # best model from inner fold and average over outer fold
    compete_from_keys = ['seed'] + additional_keys.setdefault('compete_from_keys', []) # the keys to pick the best model instance
    filter_dict = {'agent_type': agent_type}
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict, inner_fold_perf_key=inner_fold_perf_key)
    perf['hidden_dim'] = perf.apply(lambda row: cog_hidden_dim[row['cog_type']], axis=1)
    summary['hidden_dim'] = summary.apply(lambda row: cog_hidden_dim[row['cog_type']], axis=1)
    if verbose:
        with pd_full_print_context():
            print(perf)
    return perf, summary


def find_best_models_for_exp(exp_folder, cog_agent_type, additional_rnn_keys=None, additional_cog_keys=None, has_rnn=True, has_cog=True):
    goto_root_dir.run()
    check_missing_models(exp_folder)

    # for inner_fold_perf_key in ['trainval_loss', #'test_loss']:
    inner_fold_perf_key = 'trainval_loss'
    if inner_fold_perf_key == 'trainval_loss':
        fname = ''
    else:
        fname = '_based_on_test'
    print('========Select best models based on inner_fold_perf_key:', inner_fold_perf_key,'exp_folder:', exp_folder)
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)
    if has_rnn:
        for combine_model_then_select in [None]:#, {'rnn_type': ['GRU', 'SGRU'], 'finetune':[True, False, 'none']}]:
            rnn_perf, rnn_summary, perf_est_dim = select_final_rnn_perf(exp_folder, additional_keys=additional_rnn_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                                        return_dim_est=True, combine_model_then_select=combine_model_then_select)
            fname_temp = fname
            if combine_model_then_select is not None:
                fname_temp += '_combine_then_select'
            joblib.dump(rnn_perf, ana_exp_path / f'rnn_final_perf{fname_temp}.pkl')
            joblib.dump(rnn_summary, ana_exp_path / f'rnn_final_best_summary{fname_temp}.pkl')
            joblib.dump(perf_est_dim, ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.pkl')
    else:
        empty_pd = pd.DataFrame()
        joblib.dump(empty_pd, ana_exp_path / f'rnn_final_perf{fname}.pkl')
        joblib.dump(empty_pd, ana_exp_path / f'rnn_final_best_summary{fname}.pkl')

    if has_cog:
        cog_perf, cog_summary = select_final_cog_perf(exp_folder, cog_agent_type, additional_keys=additional_cog_keys, inner_fold_perf_key=inner_fold_perf_key)
        joblib.dump(cog_perf, ana_exp_path / f'cog_final_perf{fname}.pkl')
        joblib.dump(cog_summary, ana_exp_path / f'cog_final_best_summary{fname}.pkl')
    else:
        empty_pd = pd.DataFrame()
        joblib.dump(empty_pd, ana_exp_path / f'cog_final_perf{fname}.pkl')
        joblib.dump(empty_pd, ana_exp_path / f'cog_final_best_summary{fname}.pkl')


def compile_perf_for_exps(exp_folders, compile_exp_folder, additional_rnn_keys=None, rnn_filter=None, additional_cog_keys=None, additional_rnn_agg=None, additional_cog_agg=None, has_rnn=True, has_cog=True):
    rnn_perf_list = []
    cog_perf_list = []
    additional_rnn_keys = additional_rnn_keys or {}
    additional_cog_keys = additional_cog_keys or {}
    additional_rnn_agg = additional_rnn_agg or {}
    additional_cog_agg = additional_cog_agg or {}
    rnn_filter = rnn_filter or {}
    for exp_folder in exp_folders:
        ana_exp_path = ANA_SAVE_PATH / exp_folder
        if has_cog:
            cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
            cog_perf['exp_folder'] = exp_folder
            cog_perf_list.append(cog_perf)
        if has_rnn:
            rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
            rnn_perf['exp_folder'] = exp_folder
            for k, v in rnn_filter.items():
                rnn_perf = rnn_perf[rnn_perf[k] == v]
            rnn_perf_list.append(rnn_perf)
    rnn_perf = pd.concat(rnn_perf_list) if has_rnn else pd.DataFrame()
    cog_perf = pd.concat(cog_perf_list) if has_cog else pd.DataFrame()

    if has_rnn:
        model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC'] + additional_rnn_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        rnn_perf = rnn_perf.groupby(model_identifier_keys, as_index=False).agg(
            test_loss=('test_loss','mean'),
            sub_count=('test_loss','count'),
            **additional_rnn_agg,
        )
        with pd_full_print_context():
            print(rnn_perf)
    if has_cog:
        model_identifier_keys = ['cog_type'] + additional_cog_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        cog_perf = cog_perf.groupby(model_identifier_keys, as_index=False).agg(
            test_loss=('test_loss','mean'),
            hidden_dim=('hidden_dim','max'),
            sub_count=('test_loss', 'count'),
            **additional_cog_agg,
        )
        with pd_full_print_context():
            print(cog_perf)

    L = []
    if has_rnn:
        for exp_folder in exp_folders:
            ana_exp_path = ANA_SAVE_PATH / exp_folder
            perf_est_dim = joblib.load(ana_exp_path / 'rnn_final_perf_est_dim.pkl')
            perf_est_dim = perf_est_dim[perf_est_dim['less_than_former'] == 1]
            dim = perf_est_dim['hidden_dim'].max()
            L.append({'exp_folder': exp_folder, 'dimension': dim})
    df = pd.DataFrame(L)
    with pd_full_print_context():
        print(df)

    ana_exp_path = ANA_SAVE_PATH / compile_exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)
    joblib.dump(rnn_perf, ana_exp_path / 'rnn_final_perf.pkl')
    joblib.dump(df, ana_exp_path / 'rnn_final_perf_est_dim.pkl')
    joblib.dump(cog_perf, ana_exp_path / 'cog_final_perf.pkl')