from plotting import *
from plotting_dynamics import *

save_pdf = True
plotting_pipeline = [
    # 'plot_model_perf_for_each_exp',
    'plot_perf_for_all_exps',
    # 'plot_dim_for_all_exps',
    # 'plot_model_num_par',
    # 'plot_ev_for_each_exp',
    # 'plot_model_perf_data_proportion_for_each_exp',
    # 'plot_model_neuron_decoding_perf_for_each_exp',
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
    '2d_vector_field',
    ]
exp_folders = [
    # 'exp_seg_akamratrts279',
    # 'exp_seg_akamratrts278',
    # 'exp_seg_akamratrts277',
    # 'exp_seg_akamratrts275',
    # 'exp_seg_akamratrts274',
    # 'exp_seg_akamratrts273',
    # 'exp_seg_akamratrts272',
    'exp_seg_akamratrts271',
    # 'exp_seg_akamratrts270',
    # 'exp_seg_akamratrts269',
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
model_curve_setting = {
    'GRU+SGRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU': ModelCurve('SGRU', 'SGRU', SGRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'PNR1': ModelCurve('SLIN', 'SLIN', PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MFs': ModelCurve('MFs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0s': ModelCurve('MBs', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MBsvlr': ModelCurve('MBsvlr', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MBsflr': ModelCurve('MBsflr', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),

    'MB0se': ModelCurve('MB0se', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'LS0': ModelCurve('LS0', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'LS1': ModelCurve('LS1', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'MB0': ModelCurve('MB0', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MB1': ModelCurve('MB1', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MB0m': ModelCurve('MB0m', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'MB0md': ModelCurve('MB-GRU', 'MB', MB_color, dot_alpha, 'D', markersize, 1, '-'),
    'RC': ModelCurve('RC', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(0)': ModelCurve('Q(0)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(1)': ModelCurve('Q(1)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
}


plot_perf_exp_folders = exp_folders if 'plot_model_perf_for_each_exp' in plotting_pipeline else []
plot_perf_exp_folders += ['exp_seg_akamratrts'] if 'plot_perf_for_all_exps' in plotting_pipeline else []
for exp_folder in plot_perf_exp_folders:
    for add_text in [True, False]:
        plot_all_model_losses(exp_folder,
                          rnn_types=[#'GRU', 'SGRU', #'PNR1'
                              'GRU+SGRU',
                                     ],
                          cog_types=['MFs', 'MB0s',
                                     'MB0', 'MB1', #'MBsvlr','MBsflr',#'MB0md',
                                     'RC',
                                     'Q(0)', 'Q(1)'],
                          rnn_filters={'readout_FC': True},
                          xlim=[0.91, 22],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          ylim=[0.3, 0.61],
                          yticks=[0.3, 0.4, 0.5, 0.6],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          title=exp_folder[8:],
                          figname='loss_all_models',
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )

if 'plot_dim_for_all_exps' in plotting_pipeline:
    plot_dim_distribution('exp_seg_akamratrts')

if 'plot_ev_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf, plot_ev=True)

if 'plot_model_num_par' in plotting_pipeline:
    for add_text in [True, False]:
        exp_folder = exp_folders[0]
        plot_all_model_losses(exp_folder,
                          rnn_types=['GRU', 'SGRU', #'PNR1'
                                     ],
                          cog_types=['MFs', 'MB0s',  'MB0', 'MB1', #'MB0md',
                                     'RC',
                                     'Q(0)', 'Q(1)'],
                          xlim=[0.91, 22],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          perf_type='num_params',
                          title=exp_folder[8:],
                          figname='num_params_all_models',
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          )


# dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline, save_pdf=save_pdf)

if 'plot_1d_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        # plot_1d_logit_coef(exp_folder)
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='intercept')
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='slope')

