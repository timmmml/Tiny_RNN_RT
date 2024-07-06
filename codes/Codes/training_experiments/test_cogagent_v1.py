import numpy as np
import importlib
from utils import goto_root_dir
from agents import RTS_RL_agents
from datasets import Dataset

class CTSO_session:
    pass
if __name__ == '__main__':
    # session = {
    #     'choices': np.random.randint(0, 2, 1000),
    #     'second_steps': np.random.randint(0, 2, 1000),
    #     'outcomes': np.random.randint(0, 2, 1000),
    #     'n_trials': 1000,
    # }
    # CTSO_session.CTSO = session
    # CTSO_session.n_trials = session['n_trials']
    behav_dt = Dataset('MillerRat',
            behav_data_spec={'animal_name': 'm64', #'m64',#'m55'
                             }).behav_to({'behav_format':'cog_session'})
    behav_data = behav_dt.get_behav_data(np.arange(behav_dt.batch_size), {'behav_format': 'cog_session'})
    CTSO_session = behav_data['input'][0]
    session = CTSO_session.CTSO
    session['n_trials'] = CTSO_session.n_trials
    run_v0 = False
    for model_name in [
        # 'BAS',
        # 'Model_free_symm',
        # 'Model_based_symm_varlr',
        'Model_based_symm_acthist',
        # 'Q0',
        # 'Q1',
        # 'Reward_as_cue',
        # 'Model_based',
        # 'Model_based_decay',
        # 'Model_based_symm',
        # 'Model_based_mix_decay',
        # 'Model_based_mix',
        # 'Latent_state_softmax',
        # 'Latent_state_softmax_bias',
    ]:
        if run_v0: agent_v0 = getattr(RTS_RL_agents, model_name)()
        agent_type_class = importlib.import_module('agents.RTS_RL_agents_v1.'+model_name)
        agent_v1 = getattr(agent_type_class, model_name)()
        print(model_name)
        if run_v0:
            assert agent_v0.name == agent_v1.name, (agent_v0.name, agent_v1.name)
            assert agent_v0.param_names == agent_v1.param_names, (agent_v0.param_names, agent_v1.param_names)
            assert agent_v0.params == agent_v1.params, (agent_v0.params, agent_v1.params)
            assert agent_v0.param_ranges == agent_v1.param_ranges, (agent_v0.param_ranges, agent_v1.param_ranges)
            assert agent_v0.n_params == agent_v1.n_params, (agent_v0.n_params, agent_v1.n_params)

        if run_v0: DV_v0 = agent_v0.session_likelihood(CTSO_session, agent_v0.params, get_DVs=True)
        DV_v1 = agent_v1.session_likelihood(session, agent_v1.params, get_DVs=True)
        if run_v0:
            for key in DV_v0.keys():
                print(key)
                assert np.allclose(DV_v0[key], DV_v1[key]), (key, DV_v0[key], DV_v1[key])

    # print(DV_v1['Q_td_f'] - DV['Q_td_f'])
    # print(DV_v1['Q_td_s'] - DV['Q_td_s'])
    # print(DV_v1['choice_probs'] - DV['choice_probs'])
    # print(DV_v1['scores'] - DV['scores'])
    # print(DV_v1['session_log_likelihood'] - DV['session_log_likelihood'])
    # from tasks import akam_tasks as ts
    # task = ts.Two_step(com_prob=0.8, rew_gen='blocks',
    #                    block_length=50, probs=[0.2, 0.8])
    # ag_v1.simulate(task, 100)

