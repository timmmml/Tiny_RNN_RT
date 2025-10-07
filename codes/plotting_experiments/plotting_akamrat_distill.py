from plotting import *
from plotting_dynamics import *

save_pdf = True
plotting_pipeline = [
    'plot_model_perf_for_each_exp',
    # 'embedding_correlation'
]
exp_folders = [
    'exp_seg_akamrat49_distill',
]

def plot_all_model_losses(exp_folder, xlim=None, ylim=None,  xticks=None, yticks=None,
                          max_hidden_dim=20, minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models',
                          model_curve_setting=None, add_text=False, save_pdf=True):

    if figsize is None:
        figsize = (1.5, 1.5)

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    fig, ax = plot_start(figsize=figsize)

    rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    # plot_figure = 'hidden_dim'
    # plot_figure = 'trainval_size'
    # plot_figure = 'teacher_prop'
    plot_figure = 'final_comparison'
    # with pd_full_print_context():
    #     print(rnn_perf)
    if plot_figure == 'hidden_dim':
        rnn_perf = rnn_perf[(rnn_perf['trainval_size'] == 70) & (rnn_perf['distill'] != 'none')]
        hidden_dims = pd.unique(rnn_perf['hidden_dim'])
        print(hidden_dims)

        for hidden_dim in hidden_dims:# [100,200,400]:
            this_rnn_perf = rnn_perf[rnn_perf['hidden_dim'] == hidden_dim]
            embedding_dim = this_rnn_perf['embedding_dim']
            include_embedding = this_rnn_perf['include_embedding']
            embedding_dim *= include_embedding # set to 0 if not include_embedding
            perf = this_rnn_perf[perf_type]
            print(perf_type, hidden_dim,np.array(embedding_dim), np.array(perf))
            if len(perf) == 1:
                plt.scatter(embedding_dim, perf, label=f'GRU({hidden_dim})')
            else:
                plt.plot(embedding_dim, perf, label=f'GRU({hidden_dim})')
        plt.xlabel('# Embedding dimensions')
    elif plot_figure == 'trainval_size':
        this_rnn_perf = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] != 'none')]
        train_trial_num = this_rnn_perf['train_trial_num']
        perf = this_rnn_perf[perf_type]
        plt.plot(train_trial_num, perf, label=f'GRU({4})')

        rnn_perf = rnn_perf[rnn_perf['hidden_dim'] == 20]
        embedding_dims = pd.unique(rnn_perf['embedding_dim'] * rnn_perf['include_embedding'])
        print(embedding_dims)
        for embedding_dim in embedding_dims[5:]:
            if embedding_dim == 0:
                this_rnn_perf = rnn_perf[rnn_perf['include_embedding'] == False]
            else:
                this_rnn_perf = rnn_perf[(rnn_perf['embedding_dim'] == embedding_dim) & (rnn_perf['include_embedding'] == True)]
            trainval_size = this_rnn_perf['trainval_size']
            perf = this_rnn_perf[perf_type]
            plt.plot(train_trial_num, perf, label=f'GRU(ebd{embedding_dim})')
        plt.xlabel('# training trials from this subject')
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000], rotation=90)
    elif plot_figure == 'teacher_prop':
        rnn_perf_teacher = rnn_perf[(rnn_perf['hidden_dim'] == 20) & (rnn_perf['embedding_dim'] == 8) & (rnn_perf['include_embedding'] == True)]
        rnn_perf_none = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'none')]
        rnn_perf_student = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'student')]
        train_trial_num = rnn_perf_none['train_trial_num']
        plt.plot(train_trial_num, rnn_perf_teacher[perf_type], label=f'Teacher GRU(20)')
        plt.plot(train_trial_num, rnn_perf_none[perf_type], label=f'Ori GRU(4)')
        for teacher_prop in pd.unique(rnn_perf_student['teacher_prop']):
            this_rnn_perf_student = rnn_perf_student[rnn_perf_student['teacher_prop'] == teacher_prop]
            plt.plot(train_trial_num, this_rnn_perf_student[perf_type], label=f'Student GRU(4) {teacher_prop}')
        plt.xlabel('# training trials from this subject')
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000], rotation=90)

    elif plot_figure == 'final_comparison':
        cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')
        cog_models = pd.unique(cog_perf['cog_type'])
        assert len(cog_models) == 1 # for now
        for cog_type in cog_models:
            this_cog_perf = cog_perf[cog_perf['cog_type'] == cog_type]
            train_trial_num = this_cog_perf['train_trial_num']
            #print(this_cog_perf['test_loss'])
            plt.plot(train_trial_num, this_cog_perf['test_loss'], label=cog_type)

        rnn_perf_teacher = rnn_perf[(rnn_perf['hidden_dim'] == 20) & (rnn_perf['embedding_dim'] == 8) & (rnn_perf['include_embedding'] == True)]
        rnn_perf_none = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'none')]
        rnn_perf_student = rnn_perf[(rnn_perf['hidden_dim'] == 4) & (rnn_perf['distill'] == 'student')]
        trainval_size = rnn_perf_teacher['trainval_size']
        train_trial_num = rnn_perf_none['train_trial_num']
        plt.plot(train_trial_num, rnn_perf_teacher[perf_type], label=f'Teacher GRU(20)')
        plt.plot(train_trial_num, rnn_perf_none[perf_type], label=f'Ori GRU(4)')
        plt.plot(train_trial_num, rnn_perf_student[perf_type], label=f'Student GRU(4)')
        plt.xlabel('# training trials from this subject')
        plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000], rotation=90)

    plt.ylabel(f'Negative log likelihood ({perf_type})')
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('Hidden units')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    if add_text:
        figname = figname + '_text'
    figname = f'{figname}_{plot_figure}_{perf_type}' + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()

if 'plot_model_perf_for_each_exp' in plotting_pipeline:
    plot_all_model_losses(exp_folders[0], perf_type='test_loss')
    plot_all_model_losses(exp_folders[0], perf_type='train_loss')
    plot_all_model_losses(exp_folders[0], perf_type='val_loss')


if 'embedding_correlation' in plotting_pipeline:
    path = "exp_Lai\\rnn_type-GRU.hidden_dim-50.l1_weight-1e-05.include_embedding-True.embedding_dim-1\\outerfold0_innerfold0_seed0"
    ag = transform_model_format(path, source='path', target='agent')

    for name, x in ag.model.named_parameters():
        if name == 'embedding.weight':
            print(name, x.shape)
            embedding = x
            break
    embedding = embedding.detach().cpu().numpy()
    # plt.scatter(embedding[:, 0], embedding[:, 1])
    # plt.show()

    dt = Dataset('LaiHuman',behav_data_spec={'group':[0,1]})
    sub_bias = dt.sub_bias
    # calculate the correlation between each embedding column and the subject bias column
    from scipy.stats import pearsonr
    for i in range(embedding.shape[1]):
        for j in range(sub_bias.shape[1]):
            r, p = pearsonr(embedding[:, i], sub_bias[:, j])
            if p < 0.05:
                print('embedding column %d, subject bias column %d, r=%.3f, p=%f' % (i, j, r, p))
    # calculate the correlation between each embedding column and the subject bias column mean
    for i in range(embedding.shape[1]):
        import seaborn as sns
        from scipy.stats import pearsonr
        fig, ax = plot_start()
        x=embedding[:, i]
        y=sub_bias.mean(axis=1)
        (r, p) = pearsonr(x, y)
        if p< 0.001: # scientific notation
            p = f'{p:.3e}'
        else:
            p = f'{p:.3f}'
        sns.regplot(x=x, y=y, fit_reg=True, label=r'$\rho$'+f'={r:.2f}, p={p}', scatter_kws={'s': 1})
        plt.xlabel('Embedding dimension %d' % i)
        # plt.xlim([0, 0.5])
        # plt.xticks([0, 0.5])
        plt.ylabel('Subject bias')
        plt.legend()
        plt.savefig(FIG_PATH / 'exp_Lai' / f'embedding_{i}_bias_corr.pdf', bbox_inches="tight")
        plt.close()