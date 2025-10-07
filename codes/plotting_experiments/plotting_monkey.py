from plotting import *
from plotting_dynamics import *
from plotting_decoding import *

save_pdf = True
plotting_pipeline = [
    # 'plot_model_perf_for_each_exp',
    # 'plot_perf_for_all_exps',
    # 'plot_dim_for_all_exps',
    # 'plot_model_num_par',
    # 'plot_ev_for_each_exp',
    # 'plot_model_perf_data_proportion_for_each_exp',
    # 'plot_model_neuron_decoding_perf_for_each_exp',
    # 'plot_dynamics_for_each_exp',
    #'plot_1d_for_each_exp',
    # 'neural_pca_decoding_for_each_model',
    # 'decay_to_other_schematic',
    'logit_change_schematic',
]
dynamics_plot_pipeline = [
    ## global options
    # 'relative_action', # note this option will change all results for 2d_logit_ and 2d_pr_ to relative action
    # 'hist', # note this option will change all results for 2d_logit_ and 2d_pr_ to histogram
    # 'show_curve', # show curve instead of dots; only for 1d models
    # 'legend', # show legend; only for 2d_logit_change and show_curve

    ## logit and pr analysis
    # '2d_logit_change', # logit vs logit change
    # '2d_logit_next', # logit vs logit next
    # '2d_logit_nextpr', # logit vs pr next
    # '2d_pr_nextpr', # pr vs pr next
    # '2d_pr_change', # pr vs pr change
    # '2d_logit_nextpr_ci', # logit vs pr next with confidence interval; only for 1d models
    # '2d_logit_nextpr_ci_log_odds_ratio', # logit vs pr next, with log odds ratio calculated for confidence interval; only for 1d models

    ## other analysis
    # '2d_value_change',
    '2d_vector_field',
    ]
exp_folders = [
    'exp_monkeyV',
    'exp_monkeyW',
]

goto_root_dir.run()

dot_alpha = 0.9
curve_alpha= 0.9
markersize = 10
curve_markersize = 5
GRU_color = 'C0'
SGRU_color =  'C5'
LS_color = 'C1'
MF_color = 'C4'
MB_color = 'C3'
PNR_color = 'C2'
model_curve_setting = { # for monkey, all MBs are MFs
    'GRU+SGRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU': ModelCurve('SGRU', 'SGRU', SGRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU-finetune': ModelCurve('SGRU-f', 'SGRU-f',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'PNR1': ModelCurve('SLIN', 'SLIN',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MFs': ModelCurve('MFs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0s': ModelCurve('MBs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0se': ModelCurve('MB0se', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'LS0': ModelCurve('LS0', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'LS1': ModelCurve('LS1', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'MB0': ModelCurve('MB0', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB1': ModelCurve('MB1', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0m': ModelCurve('MB0m', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB0md': ModelCurve('MB-GRU', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'RC': ModelCurve('RC', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(0)': ModelCurve('Q(0)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(1)': ModelCurve('Q(1)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
}



plot_perf_exp_folders = exp_folders if 'plot_model_perf_for_each_exp' in plotting_pipeline else []
plot_perf_exp_folders += ['exp_monkey'] if 'plot_perf_for_all_exps' in plotting_pipeline else []
for exp_folder in plot_perf_exp_folders:
    # perf
    for add_text in [True, False]:
        plot_all_model_losses(exp_folder,
                          rnn_types=[#'GRU', 'SGRU',
                                     #'SGRU-finetune',
                                     #'PNR1'
                                     'GRU+SGRU',
                                     ],
                          cog_types=['MB0s', 'LS0', #'LS1',
                                     'MB0', 'MB1', #'MB0md',#'MB0m',
                                     'RC'],
                          rnn_filters={'readout_FC': True, 'finetune': 'none',
                                       },
                          xlim=[0.91, 22],
                          ylim=[0.4, 0.55],
                          yticks=[0.4, 0.5],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          title=exp_folder[4:],
                          figname='loss_all_models',
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )

if 'plot_model_num_par' in plotting_pipeline:
    for add_text in [True, False]:
        exp_folder = exp_folders[0]
        plot_all_model_losses(exp_folder,
                          rnn_types=['GRU', 'SGRU', #'PNR1'
                                     ],
                          cog_types=['MB0s', 'LS0', #'LS1',
                                     'MB0', 'MB1', #'MB0md',#'MB0m',
                                     'RC'],
                          xlim=[0.91, 22],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          perf_type='num_params',
                          title=exp_folder[4:],
                          figname='num_params_all_models',
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )

if 'plot_dim_for_all_exps' in plotting_pipeline:
    plot_dim_distribution('exp_monkey')

if 'plot_ev_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf, plot_ev=True)

if 'plot_model_neuron_decoding_perf_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        # neuron decoding
        plot_all_model_losses(exp_folder,
                          rnn_types=['GRU', 'SGRU', #'PNR1'
                                     ],
                          cog_types=['MB0s', 'LS0', 'LS1', 'MB0', 'MB1', #'MB0md','MB0m',
                                     #'RC'
                                     ],
                          rnn_filters={'readout_FC': True},
                          xlim=[0.91, 22],
                          # ylim=[0, 0.1],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          perf_type='population_R2',
                          title=exp_folder[4:],
                          figname='population_R2_all_models',
                          )

    # dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline,
         save_pdf=save_pdf)

if 'plot_1d_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        # plot_1d_logit_coef(exp_folder)
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='intercept')
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='slope')

if 'neural_pca_decoding_for_each_model' in plotting_pipeline:
    plot_neuronal_r2('exp_monkeyV', {'block_type': 'where'},
                             session_names=['V20161005', 'V20160929', 'V20160930', 'V20161017'])
    # plot_neuronal_r2('exp_monkeyW', {'block_type': 'where'},
    #                          session_names=['W20160112', 'W20160113', 'W20160121', 'W20160122'])
    # model_decoding_perf_sessions_V = plot_neural_pca_decoding('exp_monkeyV', {'block_type': 'where'}, session_names=['V20161005','V20160929','V20160930','V20161017'])
    # model_decoding_perf_sessions_W = plot_neural_pca_decoding('exp_monkeyW', {'block_type': 'where'}, session_names=['W20160112','W20160113','W20160121','W20160122'])
    # model_decoding_perf_sessions = {k: v + model_decoding_perf_sessions_W[k] for k, v in model_decoding_perf_sessions_V.items()}
    # session_names=['V20161005','V20160929','V20160930','V20161017']+['W20160112','W20160113','W20160121','W20160122']
    # model_decoding_perf_sessions_t = {i: [] for i in range(len(session_names))}
    #
    # idx = 0
    # plt.figure()
    # for k, v in model_decoding_perf_sessions.items():
    #     plt.scatter(np.ones(len(v)) * idx, v, color='k')
    #     for i in range(len(v)):
    #         model_decoding_perf_sessions_t[i].append(v[i])
    #     idx += 1
    # for i in range(8):
    #     plt.plot(range(idx), model_decoding_perf_sessions_t[i], label=session_names[i])
    # plt.xticks(range(idx), [f'{k}({np.mean(v):.3f})' for k, v in model_decoding_perf_sessions.items()], rotation=0)
    # plt.ylabel('Decoding accuracy')
    # leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
    # plt.show()


markersize = curve_markersize = 2.5
model_curve_setting = { # for monkey, all MBs are MFs
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU': ModelCurve('SGRU', 'SGRU', SGRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'PNR1': ModelCurve('SLIN', 'SLIN',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MFs': ModelCurve('MFs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0s': ModelCurve('MBs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0se': ModelCurve('MB0se', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'LS0': ModelCurve('LS0', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'LS1': ModelCurve('LS1', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'MB0': ModelCurve('MB0', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB1': ModelCurve('MB1', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0m': ModelCurve('MB0m', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB0md': ModelCurve('MB-GRU', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'RC': ModelCurve('RC', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(0)': ModelCurve('Q(0)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(1)': ModelCurve('Q(1)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
}
if 'plot_model_perf_data_proportion_for_each_exp' in plotting_pipeline:
    exp_folders_dataprop = [
    'exp_monkeyV_dataprop',
    'exp_monkeyW_dataprop',
    ]
    for exp_folder in exp_folders_dataprop:
        plot_all_model_losses_dataprop(exp_folder,
                              rnn_types=['GRU',
                                  # 'SGRU'
                                         ],
                              cog_types=[#'MB0s', 'LS0', #'LS1',
                                         'MB0', 'MB1', #'MB0md','MB0m',
                                     #'RC'
                              ],
                              xlim=[0, 5000],
                              xticks=[0, 2000,4000],
                              minorticks=False,
                              figsize=(1.5, 1.5),
                              legend=True,
                              title=exp_folder.replace('exp_seg_', ''),
                              figname='loss_all_models_dataprop',
                              model_curve_setting=model_curve_setting,
                          save_pdf=save_pdf,
                              )
if 'decay_to_other_schematic' in plotting_pipeline:
    fig, ax = plot_start(ticks_pos=False)
    # 'S0R0': (0, 0, 'A1', 'cornflowerblue'),
    # 'S0R1': (0, 1, 'A1', 'mediumblue'),
    # 'S1R0': (1, 0, 'A2', 'silver'),
    # 'S1R1': (1, 1, 'A2', 'dimgrey'),

    a, b= 0.4, 0.8
    arr_len = 0.2
    shift = 0.02
    scatter_kwargs = dict(s=30)
    quiver_kwargs = dict(angles='xy', scale_units='xy', scale=1, alpha=1, width=0.02, headwidth=5, headlength=5)
    plt.plot([0,1],[0,1], color='darkorange', linestyle='--')
    plt.scatter(a,b, c='cornflowerblue',**scatter_kwargs) # A1R0, disagree
    ax.quiver(a, b, -arr_len, 0, color=MF_color, label='Decay-to-zero', **quiver_kwargs)
    ax.quiver(a, b, arr_len, 0, color=GRU_color, label='Decay-to-the-other', **quiver_kwargs)

    c, d= 0.6, 0.2
    plt.title('A1 R=0')
    plt.scatter(c,d, c='cornflowerblue',**scatter_kwargs) # A1R0, agree
    ax.quiver(c, d, -arr_len, shift, color=MF_color, **quiver_kwargs)
    ax.quiver(c, d, -arr_len, -shift, color=GRU_color, **quiver_kwargs)

    plt.xlabel('Q(A1)')
    plt.ylabel('Q(A2)')
    plt.legend(loc='upper right')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.savefig(FIG_PATH / 'exp_monkey' / ('decay-to-other A1 R=0.pdf'), bbox_inches="tight")
    plt.show()
    plt.close()

    fig, ax = plot_start(ticks_pos=False)
    plt.plot([0,1],[0,1], color='darkorange', linestyle='--')
    plt.scatter(b,a, c='silver',**scatter_kwargs) # A2R0, disagree
    ax.quiver(b, a, 0, -arr_len, color=MF_color, label='Decay-to-zero', **quiver_kwargs)
    ax.quiver(b, a, 0, arr_len, color=GRU_color, label='Decay-to-the-other', **quiver_kwargs)

    plt.title('A2 R=0')
    plt.scatter(d,c, c='silver',**scatter_kwargs) # A2R0, agree
    ax.quiver(d, c, shift, -arr_len, color=MF_color, **quiver_kwargs)
    ax.quiver(d, c, -shift, -arr_len, color=GRU_color, **quiver_kwargs)
    plt.xlabel('Q(A1)')
    plt.ylabel('Q(A2)')
    plt.legend(loc='upper right')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.savefig(FIG_PATH / 'exp_monkey' / ('decay-to-other A2 R=0.pdf'), bbox_inches="tight")
    plt.show()
    plt.close()

if 'logit_change_schematic' in plotting_pipeline:
    from matplotlib import patches
    fig, ax = plot_start()
    color = 'cornflowerblue'
    s = 20
    ax.plot([-4,4], [2,-2], color=color, alpha=0.7)
    # line function
    line = lambda x: -0.5*x
    plt.hlines(0, -4, 4, 'darkorange', alpha=0.8, linewidth=0.4,zorder=1)
    plt.vlines(0, -4, 4, 'darkorange', alpha=0.8, linewidth=0.4, zorder=1)
    angle = 0
    for x_state in [-3.6, 3.6]:
        last_step = 2
        for step in range(last_step+1):
            plt.scatter(x_state, line(x_state), s=s, zorder=2, facecolor=color, edgecolor='k', linewidth=0.8)
            plt.scatter(x_state, 0, s=s, zorder=2, facecolor='none', edgecolor='k', linewidth=0.8)
            plt.vlines(x_state, 0, line(x_state), color='k', linewidth=1, zorder=3, alpha=0.5) #,linestyles='dotted'
            if step == last_step:
                break
            radius = np.abs(line(x_state)) * 2
            ax.quiver(x_state, 0, line(x_state), 0, angles='xy', scale_units='xy', scale=1, alpha=0.5, width=0.01, headwidth=5, headlength=5, zorder=3,color='k')
            a1=patches.Arc((x_state, 0), radius, radius,
                            angle=angle, linewidth=1, fill=False, zorder=3, theta1=0, theta2=90, color='k', alpha=0.5) # , linestyle='dotted'
            ax.add_patch(a1)
            x_state += line(x_state)
        angle += 180
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xticks([-4,0,4])
    plt.yticks([-4,0,4])
    plt.xlabel('Logit')
    plt.ylabel('Logit change')
    plt.savefig(FIG_PATH / 'exp_monkey' / 'logit_change_schematic.pdf', bbox_inches="tight")
    plt.show()
    plt.close()
