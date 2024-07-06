from plotting import *
from plotting_dynamics import *

goto_root_dir.run()
save_pdf = True
plotting_pipeline = [
    'plot_model_perf_for_each_exp',
    # 'plot_dynamics_for_each_exp',
    # 'plot_1d_for_each_exp',
]
dynamics_plot_pipeline = [
    ## global options
    # 'relative_action', # note this option will change all results for 2d_logit_ and 2d_pr_ to relative action
    # 'hist', # note this option will change all results for 2d_logit_ and 2d_pr_ to histogram
    # 'show_curve', # show curve instead of dots; only for 1d models
    # 'legend', # show legend; only for 2d_logit_change and show_curve

    ## logit and pr analysis
    '2d_logit_change', # logit vs logit change
    # '2d_logit_next', # logit vs logit next
    # '2d_logit_nextpr', # logit vs pr next
    # '2d_pr_nextpr', # pr vs pr next
    # '2d_pr_change', # pr vs pr change
    # '2d_logit_nextpr_ci', # logit vs pr next with confidence interval; only for 1d models
    # '2d_logit_nextpr_ci_log_odds_ratio', # logit vs pr next, with log odds ratio calculated for confidence interval; only for 1d models

    ## other analysis
    # '2d_value_change',
    # '2d_vector_field',
    ]

exp_folders = [
    'exp_sim_metarl',
]

dot_alpha = 0.8
markersize = 5
model_curve_setting = {
    'GRU': ModelCurve('GRU', 'GRU', 'C0', 0.6, 'x', markersize, 1, '-'),
    'SGRU': ModelCurve('SGRU', 'SGRU', 'C1', 0.6, 'x', markersize, 1, '-'),
    'PNR1': ModelCurve('SLIN', 'SLIN', 'C2', 0.6, 'x', markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MFs': ModelCurve('MFs', 'MF', 'C4', dot_alpha, 'o', markersize, 1, '-'),
    'MB0s': ModelCurve('MBs', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'MBsvlr': ModelCurve('MBsvlr', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'MBsflr': ModelCurve('MBsflr', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),

    'MB0se': ModelCurve('MB0se', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'LS0': ModelCurve('LS0', 'LS', 'C5', dot_alpha, 'v', markersize, 1, '-'),
    'LS1': ModelCurve('LS1', 'LS', 'C5', dot_alpha, 'v', markersize, 1, '-'),
    'MB0': ModelCurve('MB0', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'MB1': ModelCurve('MB1', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'MB0m': ModelCurve('MB0m', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'MB0md': ModelCurve('MB-GRU', 'MB', 'C3', dot_alpha, 'D', markersize, 1, '-'),
    'RC': ModelCurve('RC', 'MF', 'C4', dot_alpha, 'o', markersize, 1, '-'),
    'Q(0)': ModelCurve('Q(0)', 'MF', 'C4', dot_alpha, 'o', markersize, 1, '-'),
    'Q(1)': ModelCurve('Q(1)', 'MF', 'C4', dot_alpha, 'o', markersize, 1, '-'),
}


if 'plot_model_perf_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        for sim_agent_name in [
                               'MB0s_seed0', 'LS0_seed0', #'LS1_seed0',
                            'MB0_seed0', 'MB1_seed0', #'MB0md_seed0', #'RC_seed0',
                               'Q(0)_seed0', 'Q(1)_seed0'
                               ]:
            plot_all_model_losses(exp_folder,
                          rnn_types=['GRU', 'SGRU', #
                                     'PNR1'
                                     ],
                          cog_types=['MB0s', 'LS0', #'LS1',
                                     'MB0', 'MB1', #'MB0md',
                                     #'RC',
                                     'Q(0)', 'Q(1)'
                                     ],
                          rnn_filters={'agent_name': sim_agent_name},
                          cog_filters={'agent_name': sim_agent_name},
                          xlim=[0.91, 5],
                          xticks=[1, 2, 3, 4, #5, 10, 20
                                  ],
                            #   ylim=[0.4, 0.6],
                            # yticks=[0.4, 0.5, 0.6],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=False,
                          title=f'{sim_agent_name}_fit_{exp_folder[8:]}',
                          figname=f'{sim_agent_name}_loss_all_models',
                            model_curve_setting=model_curve_setting,
                          save_pdf=True,
                          )
def coloring_mapping(trial_types):
    # A1S1R0, A1S1R1, A1S2R0, A1S2R1, A2S1R0, A2S1R1, A2S2R0, A2S2R1
    color_spec = np.array(['cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey']) # state coloring
    # color_spec = np.array(
    #     ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'silver', 'dimgrey', 'silver',
    #      'dimgrey'])  # action coloring
    colors = color_spec[trial_types]
    return colors

# dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf, plot_max_logit=12,coloring_mapping=coloring_mapping,
                                     rnn_filters={'rnn_type': 'SGRU', 'hidden_dim': 1},
                                     )
        # plot_1d_logit_coef(exp_folder)