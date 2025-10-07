from plotting import *
import numpy as np

if __name__ == '__main__':
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([80]) * 0.7
    # pr[50:] = 0.3
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(1-pr, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 80])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'reversal_learning_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([300]) * 0.8
    # pr[25:50] = 0.2
    # pr[100:120] = 0.2
    # pr[160:180] = 0.2
    # pr[240:] = 0.2
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(1-pr, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 300])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'two_stage_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    # fig, ax = plot_start(figsize=(1, 0.5))
    # pr = np.ones([300]) * 0.8
    # pr[40:100] = 0.4
    # pr[160:240] = 0.2
    # pr[240:] = 0.4
    # pr_1 = 1 - pr
    # pr_1[pr_1==0.6] = 0.4
    # plt.plot(pr, color='C3', linewidth=1)
    # plt.plot(pr_1, color='C0', linewidth=1)
    # plt.ylim([0, 1])
    # plt.yticks([0, 1])
    # plt.xticks([0, 300])
    # plt.xlabel('Trial')
    # plt.ylabel('Reward\nprob.')
    # plt.savefig(FIG_PATH / 'novel_two_stage_task.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

    fig, ax = plot_start(figsize=(1, 0.5))
    pr = np.ones([300]) * 0.8
    pr[40:] = 0.2
    pr_1 = 1 - pr
    plt.plot(pr, color='C1', linewidth=1)
    plt.plot(pr_1, color='C4', linewidth=1)
    plt.ylim([0, 1])
    plt.yticks([0, 1])
    plt.xticks([0, 300])
    plt.xlabel('Trial')
    plt.ylabel('Transition\nprob.')
    plt.savefig(FIG_PATH / 'novel_two_stage_task_transition.pdf', dpi=300, bbox_inches='tight')
    plt.show()