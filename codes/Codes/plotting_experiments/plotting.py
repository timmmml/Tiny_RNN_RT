import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=mpl.MatplotlibDeprecationWarning)
import numpy as np
import os
from adjustText import adjust_text
from analyzing_experiments.analyzing import pd_full_print_context
from utils import goto_root_dir
import pathlib
from pathlib import Path
from path_settings import *
import joblib
import pandas as pd
from collections import namedtuple

ModelCurve = namedtuple('ModelCurve', ('name', 'label', 'color', 'alpha', 'marker', 'markersize', 'linewidth', 'linestyle'))


def set_mpl():
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'arial'
    mpl.rcParams['savefig.dpi'] = 480

def plot_start(square=True,figsize=None,ticks_pos=True):
    '''
    unified plot params
    '''
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig,ax

def plot_start_scatter_hist(square=True,figsize=None,ticks_pos=True):
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(1.5, 0.8),constrained_layout=True)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # ax.set(aspect=1)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig, ax, ax_histx, ax_histy

def infer_color_from_behav(behav): #, include_action=False
    if (behav['action'] == behav['stage2']).all():
        include_action = False
    else:
        include_action = True
    if include_action:
        trial_color = np.char.add('C', (behav['action'] * 4 + behav['stage2'] * 2 + behav['reward']).astype('str'))
    else:
        trial_color = np.char.add('C', (behav['stage2'] * 2 + behav['reward']).astype('str'))
    print('Trial type count',np.unique(trial_color,return_counts=True))
    return trial_color

def control_ticks(x_range,y_range):
    if x_range is None:
        return
    # x_rg = 7
    # y_rg = 7
    plt.xlim([x_range[0]-0.1,x_range[1]+0.1])
    plt.xticks([x_range[0], 0, x_range[1]])
    plt.ylim([y_range[0]-0.1,y_range[1]+0.1])
    plt.yticks([y_range[0], 0, y_range[1]])


def plot_2d_values(x, y, trial_types, x_range=None, y_range=None, x_label='', y_label='', title='', ref_line=True,
                         ref_x=0.0, ref_y=0.0, ref_diag=False, hist=False, show_dot=True, show_curve=False,
                   coloring_mapping=None, color_spec=None, labels=None,
                   legend=False, plot_params=None):
    unique_trial_types = np.unique(trial_types)
    if coloring_mapping is not None:
        color_spec = []
    elif len(unique_trial_types) == 4:
        labels = ['A1/S1 R=0', 'A1/S1 R=1', 'A2/S2 R=0', 'A2/S2 R=1']
        # color_spec = np.array(['C0', 'C1', 'C5', 'C6'])
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey'])
    elif len(unique_trial_types) == 8:
        labels = ['A1 S1 R=0', 'A1 S1 R=1', 'A1 S2 R=0', 'A1 S2 R=1', 'A2 S1 R=0', 'A2 S1 R=1', 'A2 S2 R=0',
                  'A2 S2 R=1']
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey']) # state coloring
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'silver', 'dimgrey']) # action coloring
    if hist:
        fig, ax, ax_histx, ax_histy = plot_start_scatter_hist()
    else:
        fig, ax = plot_start()
    if plot_params is not None:
        s = plot_params['s']
        alpha = plot_params['alpha']
    else:
        s = 0.5
        alpha = 0.5
    if show_dot and coloring_mapping is None:
        ax.scatter(x, y, #color=color_spec[trial_types],
                   s=s, alpha=alpha,facecolors='none', edgecolors=color_spec[trial_types])
    elif show_dot and coloring_mapping is not None:
        ax.scatter(x, y, color=coloring_mapping(trial_types),
                   s=s, alpha=alpha)
        # add label for each color
        for i in range(len(color_spec)):
            ax.scatter([], [], color=color_spec[i], label=labels[i])
    if show_curve:
        for i in range(len(color_spec)):
            x_ = x[trial_types == i]
            y_ = y[trial_types == i]
            # x_, y_ sort by x_
            sort_idx = np.argsort(x_)
            x_ = x_[sort_idx]
            y_ = y_[sort_idx]
            ax.plot(x_, y_, color=color_spec[i], label=labels[i])
    if ref_line:
        plt.hlines(ref_y, x_range[0], x_range[1], 'darkorange', alpha=0.8, linewidth=0.4)
        plt.vlines(ref_x, y_range[0], y_range[1], 'darkorange', alpha=0.8, linewidth=0.4)
    if ref_diag:
        plt.plot([x_range[0], x_range[1]], [y_range[0], y_range[1]], 'k', alpha=0.3)

    control_ticks(x_range, y_range)

    if hist:
        binwidth = 0.05
        x_left, x_right = x.min(), x.max() + binwidth
        y_bottom, y_top = y.min(), y.max() + binwidth
        for tt in np.unique(trial_types):
            ax_histx.hist(x[trial_types == tt], bins=np.arange(x_left, x_right, binwidth), alpha=0.5, color=color_spec[tt],histtype='step')
            ax_histy.hist(y[trial_types == tt], bins=np.arange(y_bottom, y_top, binwidth), alpha=0.5, color=color_spec[tt], orientation='horizontal', histtype='step')
        for ax_hist in [ax_histx, ax_histy]:
            ax_hist.spines["right"].set_visible(False)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["bottom"].set_visible(False)
            ax_hist.spines["left"].set_visible(False)
        # remove tick and tick label
        ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_histx.set_yticks([])
        # [ax_histx.spines[loc].set_visible(False) for loc in ['top', 'left', 'right']]
        ax_histy.tick_params(axis="y", labelleft=False, left=False)
        ax_histy.set_xticks([])
        # [ax_histy.spines[loc].set_visible(False) for loc in ['top', 'bottom', 'right']]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)


def plot_perf_curves(rnn_perf, rnn_types=None, model_curves=None, max_hidden_dim=20, perf_type='test_loss'):
    if rnn_types is None:
        rnn_types = pd.unique(rnn_perf['rnn_type'])
    if model_curves is None:
        raise ValueError('model_curves must be provided')
    for rnn_type in rnn_types:
        # if 'finetune' in rnn_perf.columns:
        #     if 'finetune' in rnn_type:
        #         this_rnn_perf = rnn_perf[(rnn_perf['finetune'].isin([True])) & (rnn_perf['rnn_type'] == rnn_type.split('-')[0])]
        #     else:
        #         this_rnn_perf = rnn_perf[(rnn_perf['finetune'] == 'none') & (rnn_perf['rnn_type'] == rnn_type)]
        # else:
        #     this_rnn_perf = rnn_perf[rnn_perf['rnn_type'] == rnn_type]
        if '+' in rnn_type:
            this_rnn_perf = rnn_perf[((rnn_perf['rnn_type']=='GRU') & (rnn_perf['hidden_dim']>1)) | ((rnn_perf['rnn_type']=='SGRU') & (rnn_perf['hidden_dim']==1))]
        else:
            this_rnn_perf = rnn_perf[rnn_perf['rnn_type'] == rnn_type]
            this_rnn_perf = this_rnn_perf[this_rnn_perf['hidden_dim'] <= max_hidden_dim]
        # sort based on hidden_dim
        this_rnn_perf = this_rnn_perf.sort_values(by='hidden_dim')
        hidden_dim = this_rnn_perf['hidden_dim']
        perf = this_rnn_perf[perf_type]
        model_curve = model_curves[rnn_type]
        plt.plot(hidden_dim, perf, label=model_curve.name, color=model_curve.color, alpha=model_curve.alpha,
                 linestyle=model_curve.linestyle, linewidth=model_curve.linewidth,
                 marker=model_curve.marker, markersize=model_curve.markersize)
        # plt.errorbar(hidden_dim, test_loss, yerr=this_rnn_perf['test_loss_mean_inner_sem'], color=model_curve.color, alpha=model_curve.alpha)


def plot_perf_curves_dataprop(model_perf, agent_types=None, model_curves=None, max_hidden_dim=20, perf_type='test_loss'):
    if 'rnn_type' in model_perf.columns:
        perf_col_name = 'rnn_type'
    else:
        perf_col_name = 'cog_type'
        
    if agent_types is None:
        agent_types = pd.unique(model_perf[perf_col_name])
    if model_curves is None:
        raise ValueError('model_curves must be provided')
    with pd_full_print_context():
        print(model_perf)
    labels_done = []
    for agent_type in agent_types:
        this_perf = model_perf[model_perf[perf_col_name] == agent_type]
        this_perf = this_perf[this_perf['hidden_dim'] <= max_hidden_dim]
        # hidden_dim = this_perf['hidden_dim']
        # trainval_percent = this_perf['trainval_percent']
        train_trial_num = this_perf['mean_train_trial_num']
        perf = this_perf[perf_type]
        # remove perf > 0.69
        # train_trial_num = train_trial_num[perf < 0.7]
        # perf = perf[perf < 0.7]

        model_curve = model_curves[agent_type]
        label = model_curve.label
        if label in labels_done:
            label = None
        else:
            labels_done.append(label)
        plt.plot(train_trial_num, perf, label=label, color=model_curve.color, alpha=model_curve.alpha,
                 linestyle=model_curve.linestyle, linewidth=model_curve.linewidth,
                 marker=model_curve.marker, markersize=model_curve.markersize)

def plot_perf_dots(cog_perf, cog_types=None, cog_ignores=None, model_dots=None, perf_type='test_loss', add_text=False):
    if cog_types is None:
        cog_types = set(pd.unique(cog_perf['cog_type']))
    if cog_ignores is not None:
        cog_types = cog_types - set(cog_ignores)
    texts = []
    labels_done = []
    for cog_type in cog_types:
        this_cog_perf = cog_perf[cog_perf['cog_type'] == cog_type]
        perf = this_cog_perf[perf_type].values[0]
        hidden_dim = this_cog_perf['hidden_dim'].values[0]
        model_dot = model_dots[cog_type]
        if perf_type=='test_loss' and perf > 1:
            print(f'Warning: test loss {perf} is too large, ignored:', cog_type)
            continue
        label = model_dot.label
        if label in labels_done:
            label = None
        else:
            labels_done.append(label)
        plt.scatter(hidden_dim, perf, color=model_dot.color,
                    # facecolors = 'none', edgecolors=model_dot.color,
                    alpha=model_dot.alpha, marker=model_dot.marker,
                    s=model_dot.markersize, zorder=2, label=label)

        # plt.errorbar(hidden_dim, perf, yerr=this_cog_perf['test_loss_mean_inner_sem'].values[0], color=model_dot.color, alpha=model_dot.alpha)
        if add_text:
            texts.append(plt.text(hidden_dim, perf, model_dot.name, ha='center', va='center', fontsize=5,
                              color=model_dot.color))
    adjust_text(texts)  # , arrowprops=dict(arrowstyle='->', color='black'))


def set_brief_xlog(xlim=None, xticks=None, minorticks=False):
    if xlim is None:
        xlim = [0.91, 22] # the min value is 1, max value is 20
    if xticks is None:
        xticks = [1, 2, 3, 4, 5, 10, 20]
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # allow us to set xticks with log scale
    if not minorticks:
        plt.minorticks_off() # turn off minor ticks

def set_brief_ylog(ylim=None, yticks=None, minorticks=False):
    ax = plt.gca()
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # allow us to set yticks with log scale
    if not minorticks:
        plt.minorticks_off() # turn off minor ticks

def plot_dim_distribution(exp_folder,save_pdf=True):
    goto_root_dir.run()
    df = joblib.load(ANA_SAVE_PATH / exp_folder / 'rnn_final_perf_est_dim.pkl')
    df = df['dimension'].values
    fig, ax = plot_start(figsize=(1.5, 1.5))
    count, _, _ = plt.hist(df, bins=[1,2,3,4,5],color='white', linewidth=2, edgecolor='black')
    plt.xticks([1.5,2.5,3.5,4.5,5.5], [1,2,3,4,5])
    plt.yticks(np.arange(0, count.max()+1, 1))
    plt.xlabel('Estimated dimensionality')
    plt.ylabel('# Subjects')

    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    fname = 'est_dim' + '.pdf' if save_pdf else 'est_dim' + '.png'
    plt.savefig(fig_exp_path / fname, bbox_inches="tight")
    plt.show()

def plot_all_model_losses(exp_folder, rnn_types=None, cog_types=None, rnn_filters=None, cog_filters=None, xlim=None, ylim=None,  xticks=None, yticks=None,
                          max_hidden_dim=20, minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models',
                          model_curve_setting=None, add_text=False, save_pdf=False):
    if cog_types is None:
        raise ValueError('cog_types must be provided')
    if xlim is None:
        xlim = [0.91, 22]
    if xticks is None:
        xticks = [1, 2, 3, 4, 5, 10, 20]
    if rnn_filters is None:
        rnn_filters = {}
    if cog_filters is None:
        cog_filters = {}
    if figsize is None:
        figsize = (1.5, 1.5)
    assert model_curve_setting is not None

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    if perf_type == 'test_loss':
        rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
        # if len(rnn_types) == 1 and '+' in rnn_types[0]:
        #     rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf_combine_then_select.pkl')
        cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
        with pd_full_print_context():
            print(rnn_perf)
            print(cog_perf)

    elif perf_type in ['mean_R2', 'mean_R2_max', 'population_R2']:
        rnn_perf = joblib.load(ana_exp_path / 'rnn_neuron_decoding_perf_based_on_test.pkl')
        cog_perf = joblib.load(ana_exp_path / 'cog_neuron_decoding_perf_based_on_test.pkl')
    elif perf_type in ['num_params']:
        rnn_perf = joblib.load(ana_exp_path / 'rnn_type_num_params.pkl')
        cog_perf = joblib.load(ana_exp_path / 'cog_type_num_params.pkl')
        with pd_full_print_context():
            print(rnn_perf)
            print(cog_perf)
    else:
        raise ValueError('perf_type not recognized')
    if rnn_types is not None and len(rnn_types) > 0:
        for k, v in rnn_filters.items():
            if k in rnn_perf.columns:
                rnn_perf = rnn_perf[rnn_perf[k] == v]
    for k, v in cog_filters.items():
        cog_perf = cog_perf[cog_perf[k] == v]

    fig, ax = plot_start(figsize=figsize)
    set_brief_xlog(xlim=xlim, xticks=xticks, minorticks=minorticks)
    if perf_type in ['num_params']:
        set_brief_ylog()
    print('Plotting for', exp_folder)
    if rnn_types is not None and len(rnn_types) > 0:
        plot_perf_curves(rnn_perf, rnn_types, model_curve_setting, max_hidden_dim=max_hidden_dim, perf_type=perf_type)
    plot_perf_dots(cog_perf, cog_types, model_dots=model_curve_setting, perf_type=perf_type, add_text=add_text)
    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2' or perf_type == 'mean_R2_max':
        plt.ylabel('R2')
    elif perf_type == 'population_R2':
        plt.ylabel('Population R2')
        plt.hlines(rnn_perf.iloc[0]['population_task_R2'], xlim[0], xlim[1], color='k', linestyle='--', linewidth=1)
    elif perf_type == 'num_params':
        plt.ylabel('# Parameters')
    plt.xlabel('# Dynamical variables (d)')
    if yticks is not None:
        plt.yticks(yticks)
    if ylim is not None:
        plt.ylim(ylim)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    if add_text:
        figname = figname + '_text'
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()

def plot_all_model_losses_dataprop(exp_folder, rnn_types=None, cog_types=None, rnn_filters=None, cog_filters=None, xlim=None, xticks=None,
                          minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models_dataprop',
                                   model_curve_setting=None,
                          save_pdf=False,):
    if rnn_types is None:
        raise ValueError('rnn_types must be provided')
    if cog_types is None:
        raise ValueError('cog_types must be provided')
    if figsize is None:
        figsize = (1.5, 1.5)
    if rnn_filters is None:
        rnn_filters = {}
    if cog_filters is None:
        cog_filters = {}

    assert perf_type in ['test_loss', 'mean_R2']

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
    if rnn_types is not None and len(rnn_types) > 0:
        for k, v in rnn_filters.items():
            if k in rnn_perf.columns:
                rnn_perf = rnn_perf[rnn_perf[k] == v]
    for k, v in cog_filters.items():
        cog_perf = cog_perf[cog_perf[k] == v]

    fig, ax = plot_start(figsize=figsize)
    # set_brief_xlog(xlim=xlim, xticks=xticks, minorticks=minorticks)
    if xlim is not None:
        plt.xlim(xlim)
    if xticks is not None:
        plt.xticks(xticks)
    print('Plotting for', exp_folder)
    plot_perf_curves_dataprop(rnn_perf, rnn_types, model_curve_setting, perf_type=perf_type)
    plot_perf_curves_dataprop(cog_perf, cog_types, model_curve_setting, perf_type=perf_type)

    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2':
        plt.ylabel('R2')

    plt.xlabel('# Trials for training')
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == '__main__':
    pass