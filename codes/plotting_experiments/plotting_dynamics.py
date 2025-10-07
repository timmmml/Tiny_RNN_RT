import matplotlib.pyplot as plt

from plotting import *
from analyzing_experiments.analyzing_dynamics import *

from utils import goto_root_dir

from statsmodels.stats.contingency_tables import Table2x2
def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax=None):
    arrow_max_num = 200
    arrow_alpha = 0.8
    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx]
    ax.quiver(x1, x2, x1_change, x2_change, color=color,
              angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, width=0.004, headwidth=10, headlength=10)
    axis_min, axis_max = axis_range
    # ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--')
    # ax.plot([axis_max, axis_min], [axis_min, axis_max], 'k--')
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    # xleft, xright = my_ceil(axis_min, 1), my_floor(axis_max, 1)
    if axis_min < 0 < axis_max:
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min,1), 0, np.round(axis_max,1)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min,1), np.round(axis_max,1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_aspect('equal')

def plt_2d_vector_magnitute(x1, x1_change, x2, x2_change, change_magnitude, change_magnitude_max, ax=None, cbar=True):
    s = int(np.sqrt(len(x1)))
    X_mesh = x1.reshape((s, s))
    Y_mesh = x2.reshape((s, s))
    Z_mesh = change_magnitude.reshape((s, s)) / change_magnitude_max
    # print(X_mesh, Y_mesh, Z_mesh)
    ax.contour(X_mesh, Y_mesh, Z_mesh, levels=20, linewidths=0.5, colors='k', alpha=0.3)
    ctf = ax.contourf(X_mesh, Y_mesh, Z_mesh, levels=20, alpha=0.2, vmin=0, vmax=1)
    # cbar = plt.colorbar()
    if cbar:
        cbar = ax.get_figure().colorbar(ctf, ax=ax)
        change_magnitude_max = np.round(change_magnitude_max, 1)
        cbar.set_label('Speed of dynamics', rotation=270)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([0, change_magnitude_max])

def plt_ev(ev, trial_type,ax):
    value = ev[trial_type]['eigenvalue']
    rvector = ev[trial_type]['eigenvector']
    b = ev[trial_type]['bias']
    if value[0] > value[1]:
        value = value[::-1]
        rvector = rvector[:, ::-1]
    lvector = np.linalg.inv(rvector).T
    # d = np.linalg.inv(vector) @ b
    # d/=np.linalg.norm(d)*2
    for j in range(2):
        if np.isclose(rvector[0, j], lvector[0, j]) and np.isclose(rvector[1, j], lvector[1, j]):
            ax.plot([0, rvector[0, j]], [0, rvector[1, j]], 'w', label=f'ev{value[j]:.3f}', linewidth=1 + j * 2)
        else:
            ax.plot([0, rvector[0, j]], [0, rvector[1, j]], 'w', label=f'rev{value[j]:.3f}', linewidth=1 + j * 2)
            ax.plot([0, lvector[0, j]], [0, lvector[1, j]], 'y', label=f'lev{value[j]:.3f}', linewidth=1 + j * 2)
        # ax.plot([0, d[j]*vector[0,j]], [0, d[j]*vector[1,j]],'y', linewidth=1+j*2)
    # ax.plot([0, b[0]], [0, b[1]], 'y', label=f'b', linewidth=2)
    ax.legend()

def plt_2d_vector_field(model_true_output, model_true_trial_types, model_1step_output=None, model_1step_trial_types=None,
                        readout_vector=None, readout_bias=None, subplot=True, ev=None, title=True, coloring_mapping=None,):
    x1, x1_change = extract_value_changes(model_true_output, value_type=0)
    x2, x2_change = extract_value_changes(model_true_output, value_type=1)
    axis_max = max(np.max(x1),np.max(x2))
    axis_min = min(np.min(x1),np.min(x2))

    # color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey']) # state coloring
    color_spec = np.array(
        ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'silver',
         'dimgrey'])  # action coloring
    if len(model_true_trial_types.shape) == 1:
        unique_trial_types = np.unique(model_true_trial_types)
    elif len(model_true_trial_types.shape) == 2:
        unique_trial_types = np.array([0, 1, 2, 3])
    else:
        raise ValueError('model_true_trial_types should be a vector or a 3D array')
    if len(unique_trial_types) == 4:
        titles = ['A1/S1 R=0', 'A1/S1 R=1', 'A2/S2 R=0', 'A2/S2 R=1']
        row_num, col_num = 4, 1
        locs = [1, 2, 3, 4]
    elif len(unique_trial_types) == 8:
        titles = ['A1,S1,R=0', 'A1,S1,R=1', 'A1,S2,R=0', 'A1,S2,R=1','A2,S1,R=0', 'A2,S1,R=1', 'A2,S2,R=0', 'A2,S2,R=1']
        locs = [1, 3, 2, 4, 5, 7, 6, 8]
        row_num, col_num = 4, 2
    else:
        raise ValueError
    set_mpl()
    if subplot:
        # put all subplots in one figure
        fig = plt.figure(figsize=(4, 8))
        axes = [plt.subplot(row_num, col_num, locs[trial_type]) for trial_type in unique_trial_types]
    else:
        # put each subplot in a separate figure
        axes = [plt.figure(figsize=(2, 2)).gca() for trial_type in unique_trial_types]
    if model_1step_output is not None:
        x1_1step, x1_change_1step = extract_value_changes(model_1step_output, value_type=0)
        x2_1step, x2_change_1step = extract_value_changes(model_1step_output, value_type=1)
        change_magnitude = np.sqrt(x1_change_1step ** 2 + x2_change_1step ** 2)
        change_magnitude_max = np.max(change_magnitude)
    for trial_type in unique_trial_types:
        ax = axes[trial_type]
        if len(model_true_trial_types.shape) == 1:
            idx = model_true_trial_types == trial_type
        elif len(model_true_trial_types.shape) == 2:
            transform_trial_type = (model_true_trial_types[:, 0] * 2 + model_true_trial_types[:, 1]).astype(int)
            idx = transform_trial_type == trial_type
        if model_1step_output is not None:
            if len(model_1step_trial_types.shape) == 1:
                idx_1step = model_1step_trial_types == trial_type
            elif len(model_1step_trial_types.shape) == 2:
                transform_trial_type = (model_1step_trial_types[:, 0] * 2 + model_1step_trial_types[:, 1]).astype(int)
                idx_1step = transform_trial_type == trial_type
            # idx_1step = model_1step_trial_types == trial_type
            plt_2d_vector_magnitute(x1_1step[idx_1step], x1_change_1step[idx_1step],
                                    x2_1step[idx_1step], x2_change_1step[idx_1step],
                                    change_magnitude[idx_1step], change_magnitude_max, ax=ax, cbar=False)
        if coloring_mapping is not None:
            plt_2d_vector_flow(x1[idx], x1_change[idx], x2[idx], x2_change[idx], coloring_mapping(model_true_trial_types[idx]),
                               [axis_min, axis_max], ax=ax)
        else:
            plt_2d_vector_flow(x1[idx], x1_change[idx], x2[idx], x2_change[idx], 'k',#color_spec[trial_type],
                               [axis_min, axis_max], ax=ax)
        if title:
            ax.set_title(titles[trial_type])

        # draw readout vector
        if readout_vector is not None: # w1, w2
            x0, y0 = (axis_min + axis_max) / 2, (axis_min + axis_max) / 2
            w1, w2 = readout_vector
            ax.quiver(x0-w1/2, y0-w2/2, w1, w2, color='darkorange',#'k',
                      angles='xy', scale_units='xy', scale=1, alpha=0.6, headwidth=10, headlength=10)
            # draw decision boundary w1*x1 + w2*x2 + b = 0
            db_x1 = np.linspace(axis_min, axis_max, 100)
            db_x2 = -(w1 * db_x1 + readout_bias) / w2
            ax.plot(db_x1, db_x2, '--',alpha=0.7,color='darkorange',#'k'
                    )

        if ev is not None:
            plt_ev(ev, trial_type, ax)
    return axes


def infer_readout_vector(model_output, model_scores):
    from sklearn.linear_model import LinearRegression
    full_values = extract_value_changes(model_output, return_full_dim=True)[-1]
    logits = extract_value_changes(model_scores, value_type='logit')[0]
    reg = LinearRegression().fit(full_values, logits)
    readout_vector = reg.coef_ # w1, w2
    readout_bias = reg.intercept_ # b
    norm_l2 = np.linalg.norm(readout_vector)
    readout_vector /= norm_l2
    readout_bias /= norm_l2
    return readout_vector, readout_bias


def _plot_2d_value_change_whole(model_pass, fig_exp_path):
    trial_types_ori = model_pass['trial_type']
    trial_types = np.concatenate(trial_types_ori)
    hid_state_lb = model_pass['hid_state_lb']
    hid_state_ub = model_pass['hid_state_ub']
    model_output = model_pass['internal']
    hid_state_rg = hid_state_ub - hid_state_lb
    for d in range(2):
        values, values_change = extract_value_changes(model_output, value_type=d)
        plot_2d_values(values, values_change, trial_types,
                       x_range=(hid_state_lb[d], hid_state_ub[d]), y_range=(-hid_state_rg[d], hid_state_rg[d]),
                       x_label=f'{d + 1} Value', y_label=f'{d + 1} Value change', title='',
                       ref_line=True)
        plt.savefig(fig_exp_path / ('2d_values.png'), bbox_inches="tight")
        plt.show()
        plt.close()


def _plot_2d_vector_field_whole(model_pass, model_pass_1step, fig_exp_path, ev=None, save_pdf=False, title=True, coloring_mapping=None, output_h0=True):
    trial_types_ori = model_pass['trial_type']
    if not output_h0:
        # we should remove the last time step in trial_types_ori
        # in this case, model_scores and trial_types_ori had the same length
        trial_types_ori = [trial_type[:-1] for trial_type in trial_types_ori]
    trial_types = np.concatenate(trial_types_ori)
    model_output = model_pass['internal']
    model_scores = model_pass['scores']
    if model_pass_1step is not None:
        model_1step_output = model_pass_1step['internal']
        model_1step_trial_types = np.array(model_pass_1step['trial_type']).flatten()
    else:
        model_1step_output = None
        model_1step_trial_types = None
    readout_vector, readout_bias = infer_readout_vector(model_output, model_scores)
    axes = plt_2d_vector_field(model_output, trial_types, model_1step_output=model_1step_output,
                               model_1step_trial_types=model_1step_trial_types, readout_vector=readout_vector,
                               readout_bias=readout_bias, subplot=True, ev=ev, title=title, coloring_mapping=coloring_mapping)
    figs = [ax.get_figure() for ax in axes]
    if figs[0] is figs[1]:
        fig = figs[0]
        figname = f'2d_vector_field' + ('.pdf' if save_pdf else '.png')
        fig.savefig(fig_exp_path / figname, bbox_inches="tight")
        fig.show()
        plt.close(fig)
    else:
        for i_f, fig in enumerate(figs):
            figname = f'2d_vector_field_{i_f}' + ('.pdf' if save_pdf else '.png')
            fig.savefig(fig_exp_path / figname, bbox_inches="tight")
            fig.show()
            plt.close(fig)

def plot_all_models_value_change(exp_folder, plots, save_pdf=False, plot_ev=False, plot_max_logit=5, rnn_filters=None, plot_params=None, coloring_mapping=None,output_h0=True, additional_fname=''):
    model_summary = get_model_summary(exp_folder)
    if rnn_filters is None:
        rnn_filters = {}
    for k, v in rnn_filters.items():
        model_summary = model_summary[model_summary[k] == v]
    if plot_ev:
        model_summary = model_summary[(model_summary['rnn_type'] == 'PNR1') & (model_summary['hidden_dim'] == 2)]
    for i, row in model_summary.iterrows():
        model_path = row['model_path']
        model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
        hidden_dim = row['hidden_dim']
        plot_one_model_value_change(model_path, hidden_dim, plots,
                                    save_pdf=save_pdf, plot_ev=plot_ev, plot_max_logit=plot_max_logit, plot_params=plot_params,
                                    coloring_mapping=coloring_mapping,output_h0=output_h0, additional_fname=additional_fname)

def plot_one_model_value_change(model_path, hidden_dim, plots,
                                save_pdf=False, plot_ev=False, plot_max_logit=5, plot_params=None, coloring_mapping=None,output_h0=True, additional_fname=''):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_scores = model_pass['scores']
    # model_output = model_pass['internal']
    trial_types_ori = model_pass['trial_type']
    if not output_h0:
        # we should remove the last time step in trial_types_ori
        # in this case, model_scores and trial_types_ori had the same length
        trial_types_ori = [trial_type[:-1] for trial_type in trial_types_ori]
    trial_types = np.concatenate(trial_types_ori)
    # print(model_output[0].shape, model_scores[0].shape, trial_types_ori[0].shape)
    fig_exp_path = FIG_PATH / model_path
    os.makedirs(fig_exp_path, exist_ok=True)
    print(f'{model_path} making {plots}')

    if '2d_value_change' in plots and hidden_dim == 2:
        _plot_2d_value_change_whole(model_pass, fig_exp_path)

    if '2d_vector_field' in plots and hidden_dim == 2:
        if os.path.exists(ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl'):
            model_pass_1step = joblib.load(ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl')
        else:
            model_pass_1step = None
        if plot_ev and os.path.exists(ANA_SAVE_PATH / model_path / f'eigen.pkl'):
            ev = joblib.load(ANA_SAVE_PATH / model_path / f'eigen.pkl')
        else:
            ev = None
        _plot_2d_vector_field_whole(model_pass, model_pass_1step, fig_exp_path, ev=ev, save_pdf=False, title=False,#True
                                    coloring_mapping=coloring_mapping,output_h0=output_h0)


    if '2d_logit_change' in plots or \
        '2d_logit_next' in plots or  \
        '2d_logit_nextpr' in plots or  \
        '2d_logit_nextpr_ci' in plots or  \
        '2d_pr_nextpr' in plots or \
        '2d_pr_change' in plots:
        # logit change
        logits, logits_change = extract_value_changes(model_scores, value_type='logit')
        action_prob = 1 / (1 + np.exp(-logits))
        next_logit = logits+logits_change
        next_action_prob = 1 / (1 + np.exp(-next_logit))
        prob_change = next_action_prob - action_prob

    relative_action = 'relative_action' in plots # False: absolute action, True: relative action
    if relative_action:
        if hidden_dim != 1:
            print('relative_action only works for 1d models; skipping')
            return
        action_type = trial_types > (trial_types.max() / 2)
        logits[action_type] = -logits[action_type]
        logits_change[action_type] = -logits_change[action_type]
        next_logit[action_type] = -next_logit[action_type]
        action_prob[action_type] = 1 - action_prob[action_type]
        next_action_prob[action_type] = 1 - next_action_prob[action_type]
        prob_change[action_type] = -prob_change[action_type]

    hist = 'hist' in plots
    show_curve = 'show_curve' in plots
    legend = 'legend' in plots
    fname = ('_hist' if hist else '') + ('_relaction' if relative_action else '')
    if show_curve:
        if hidden_dim != 1:
            print('show_curve only works for 1d models; skipping')
            return
        fname += '_curve'+('_legend'  if legend else '')+additional_fname+('.pdf' if save_pdf else '.png')
    else: # too many dots to save as pdf; ignore save_pdf
        fname += additional_fname+'.png'

    logit_range = (-plot_max_logit, plot_max_logit)
    if '2d_logit_change' in plots:
        plot_2d_values(logits, logits_change, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit\nPrefer A2          Prefer A1',
                       y_label='Logit change', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        if show_curve and legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        plt.savefig(fig_exp_path / (f'2d_logits{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_next' in plots:
        plot_2d_values(logits, next_logit, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit',
                       y_label='Logit (next)', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits_next{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_nextpr' in plots:
        plot_2d_values(logits, next_action_prob, trial_types, x_range=logit_range, y_range=(0, 1), x_label='Logit',
                       y_label='Action prob (next)', title='',
                       ref_line=True, ref_x=0, ref_y=0.5, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits_next_action_prob{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_pr_nextpr' in plots:
        plot_2d_values(action_prob, next_action_prob, trial_types, x_range=(0, 1), y_range=(0, 1),
                       x_label='Action prob', y_label='Action prob (next)', title='',
                       ref_line=True, ref_x=0.5, ref_y=0.5, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_action_prob_next_action_prob{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_pr_change' in plots:
        plot_2d_values(action_prob, prob_change, trial_types, x_range=(0, 1), y_range=(-1, 1), x_label='Action prob',
                       y_label='Action prob change', title='',
                       ref_line=True, ref_x=0.5, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_action_prob_prob_change{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()
    unique_trial_types = np.unique(trial_types)

    if len(unique_trial_types) == 4:
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey'])
    elif len(unique_trial_types) == 8:
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey']) # state coloring
        color_spec = np.array(
            ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'silver',
             'dimgrey'])  # action coloring
    else:
        color_spec = []

    if '2d_logit_nextpr_ci' in plots and hidden_dim == 1:
        if not show_curve:
            print('2d_logit_nextpr_ci only works with show_curve; skipping')
            return
        plot_2d_values(logits, next_action_prob, trial_types, x_range=logit_range, y_range=(0,1), x_label='Logit', y_label='Action prob (next)', title='',
                        ref_line=True, ref_x=0, ref_y=0.5, hist=hist, show_dot=False, show_curve=True, coloring_mapping=coloring_mapping)
        bin_results = extract_logit_action_freq(model_scores, trial_types_ori)

        for tt in bin_results.keys():
            bin_centers, p, ci_low, ci_upp, action_counts_of_bin = bin_results[tt]
            if relative_action and tt > (trial_types.max() / 2):
                bin_centers = -bin_centers
                p = 1 - p
                ci_low, ci_upp = 1 - ci_upp, 1 - ci_low

            plt.fill_between(bin_centers, ci_low, ci_upp, alpha=0.2, color=color_spec[tt])

        plt.savefig(fig_exp_path / (f'2d_logits_next_action_prob_CI{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_nextpr_ci_log_odds_ratio' in plots and hidden_dim == 1:
        bin_results = extract_logit_action_freq(model_scores, trial_types_ori)
        _plot_action_pair_log_odds_ratio(bin_results, fig_exp_path, save_pdf=save_pdf, color_spec=color_spec, relative_action=relative_action)

def _plot_action_pair_log_odds_ratio(bin_results, fig_exp_path, save_pdf=False, color_spec=None, relative_action=False):
    fname_prefix = 'relative_' if relative_action else ''
    if color_spec is None:
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey'])
    for tt1 in bin_results.keys():
        for tt2 in bin_results.keys():
            if tt1 >= tt2:
                continue
            bin_centers1, p1, ci_low1, ci_upp1, action_counts_of_bin1 = bin_results[tt1]
            bin_centers2, p2, ci_low2, ci_upp2, action_counts_of_bin2 = bin_results[tt2]

            plt.figure(figsize=(1.5, 3.5))
            plt.subplot(2, 1, 1)
            plt.plot(bin_centers1, p1, color=color_spec[tt1])
            plt.plot(bin_centers2, p2, color=color_spec[tt2])
            plt.vlines(0, 0, 1, color='k', alpha=0.2)
            plt.hlines(0.5, -5, 5, color='k', alpha=0.2)
            plt.xlim(-5, 5)
            plt.ylabel('Action prob')
            plt.subplot(2, 1, 2)
            log_odds_ratio = []
            lcbs = []
            ucbs = []
            for i in range(len(bin_centers1)):
                table = Table2x2([[action_counts_of_bin1[i,0], action_counts_of_bin1[i, 1]],
                                  [action_counts_of_bin2[i,0], action_counts_of_bin2[i,1]]])
                log_odds_ratio.append(table.log_oddsratio)
                lcb, ucb = table.log_oddsratio_confint(0.05, method='normal')
                lcbs.append(lcb)
                ucbs.append(ucb)
            plt.plot(bin_centers1, log_odds_ratio, color='b')
            plt.plot(bin_centers1, np.zeros(len(bin_centers1)), color='k', linestyle='--')
            plt.fill_between(bin_centers1, lcbs, ucbs, alpha=0.2, color='b')
            plt.xlim(-5, 5)
            plt.ylabel('Log odds ratio')
            plt.xlabel('Logit')
            os.makedirs(fig_exp_path / 'log_odds_ratio', exist_ok=True)
            fname = fname_prefix + f'{tt1}vs{tt2}' + ('.pdf' if save_pdf else '.png')
            plt.savefig(fig_exp_path / 'log_odds_ratio' / fname, bbox_inches="tight")
            plt.show()
            plt.close()


def plot_1d_logit_feature_simple(exp_folder, save_pdf=False, legend=True, feature='intercept'):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    if feature == 'intercept':
        group_summary = joblib.load(ana_exp_path / 'intercept_group_summary.pkl')
    elif feature == 'slope':
        group_summary = joblib.load(ana_exp_path / 'slope_group_summary.pkl')
    else:
        raise ValueError('feature must be either intercept or slope')
    ticklabels = [c for c in group_summary.columns if 4<=len(c)<=6] # column names like 'A1S1R0' or 'S1R1'
    num_trial_type = len(ticklabels)
    cond_dict = { # x location, reward color, tick label
        'A0S0R0': (0, 0, 'A1\nS1','cornflowerblue'),
        'A0S0R1': (0, 1, 'A1\nS1','mediumblue'),
        'A0S1R0': (2, 0, 'A1\nS2','cornflowerblue'),
        'A0S1R1': (2, 1, 'A1\nS2','mediumblue'),
        'A1S0R0': (3, 0, 'A2\nS1','silver'),
        'A1S0R1': (3, 1, 'A2\nS1','dimgrey'),
        'A1S1R0': (1, 0, 'A2\nS2','silver'),
        'A1S1R1': (1, 1, 'A2\nS2','dimgrey'),

        'S0R0': (0, 0, 'A1','cornflowerblue'),
        'S0R1': (0, 1, 'A1','mediumblue'),
        'S1R0': (1, 0, 'A2','silver'),
        'S1R1': (1, 1, 'A2','dimgrey'),
    }
    for i, row in group_summary.iterrows():
        model_name = row['model_type']
        if num_trial_type == 8:
            fig, ax = plot_start(figsize=(1, 1))
        else:
            fig, ax = plot_start(figsize=(0.5, 1))
        max_v = np.max([np.abs(np.mean(row[ticklabels[i]])) for i in range(num_trial_type)])
        plot_ticklabels = [''] * (num_trial_type//2)
        if feature == 'intercept':
            plt.hlines([-1, 0, 1], -0.3, num_trial_type//2 - 0.7, 'darkorange', alpha=0.4, linestyles='dashed', linewidth=0.5)
            plt.ylim([-1.3, 1.3])
            plt.yticks([-1, 0, 1])
            plt.ylabel('Asymptotic preference\nPrefer A2          Prefer A1')
        else:
            plt.ylim([-0.3, 1.3])
            plt.yticks([0, 0.5, 1])
            plt.ylabel('Learning rate')
        for i in range(num_trial_type):
            points = np.array(row[ticklabels[i]])
            if feature == 'intercept':
                points = points / max_v

            points_mean = np.mean(points)
            x, r, tick_label, color = cond_dict[ticklabels[i]]
            plot_ticklabels[x] = tick_label
            if x == 0:
                label = f'Reward {r}'
            else:
                label = None
            if r == 0:
                marker = 'o'#'s'
            else:
                marker = 'o'#'d'
            #color = ['magenta', 'green'][c]
            # plt.boxplot(points, positions=[x], widths=0.1, vert=True, showfliers=True, patch_artist=True, labels=label,
            #             boxprops=dict(facecolor=color, color=color, alpha=0.2),
            #             medianprops=dict(color=color, alpha=1),
            #             whiskerprops=dict(color=color, alpha=0.5),
            #             capprops=dict(color=color, alpha=0.5),
            #             flierprops=dict(color=color, alpha=0.5, marker='o', markersize=1)
            #
            #             )
            plt.scatter(x, points_mean, label=label, s=5, color=color, marker=marker)#, facecolors='none')
            plt.errorbar(x, points_mean, yerr=np.std(points), color=color, capsize=2)#, label=label)
            # plt.scatter(np.ones(len(points))*(x-0.1), points, s=3, color=color, alpha=0.2, marker=marker)
        plt.xticks(np.arange(num_trial_type//2), plot_ticklabels)
        plt.xlim([-0.3, num_trial_type//2-0.7])
        # plt.gca().invert_yaxis()
        # if num_trial_type == 8:
        #     plt.xlabel('Common          Rare')
        # plt.vlines(0, -1, num_trial_type, 'k', alpha=0.5)
        # flip axis

        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        plt.title(model_name)
        fig_exp_path = FIG_PATH / exp_folder
        os.makedirs(fig_exp_path, exist_ok=True)
        plt.savefig(fig_exp_path / (f'1d_logits_{feature}_{model_name}_simple'+('.pdf' if save_pdf else '.png')), bbox_inches="tight")
        plt.show()
        plt.close()
        print(f'plot {feature}: {model_name} done')