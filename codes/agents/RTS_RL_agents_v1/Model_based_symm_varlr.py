import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MB_step_core_variables(alpha_curr, alpha_other, r_rare, p_transit, c, s, o, Q, pc):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_new = Q.copy()
    # update action values.
    alpha = alpha_curr if c == pc else alpha_other
    if c == s and o == 1:
        u = 1
    elif c == s and o == 0:
        u = 0
    elif c != s and o == 1:
        u = 0
    elif c != s and o == 0:
        u = r_rare
    else:
        raise ValueError('Invalid outcome value.')
    Q_new[c] = (1. - alpha) * Q[c] + alpha * u
    Q_new[nc] = (1. - alpha) * Q[nc] - alpha * u

    pc = np.array([c])
    return Q_new, pc


@jit(nopython=True)
def _MB_session_likelihood_core(alpha_curr, alpha_other, r_rare, iTemp, p_transit, choices, second_steps, outcomes, Q, pc, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1], pc[trial + 1] = _MB_step_core_variables(alpha_curr, alpha_other, r_rare, p_transit, c, s, o, Q[trial], pc[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q[trial + 1])
    return trial_log_likelihood, Q, pc, scores, choice_probs

class Model_based_symm_varlr(TwoStepModelCoreCSO):
    def __init__(self, p_transit=0.8, equal_reward=False, fix_lr=False):
        super().__init__()
        self.name = 'Model based symm varied lr'
        self.param_names = ['alpha_curr', 'alpha_other', 'r_rare', 'iTemp']
        self.params = [0.5, 0.5, 0.5, 5.]
        self.param_ranges = ['unit', 'unit', 'unit', 'pos']
        self.n_params = 4
        self.fix_lr = fix_lr
        self.equal_reward = equal_reward
        assert not equal_reward, 'equal_reward must be True for this model.'
        self.p_transit = p_transit
        self.state_vars = ['Q']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q': np.zeros(2),
                'pc': np.zeros(1),
            }
        else:
            if 'h0' in wm:
                self.wm = {
                    'Q': wm['h0'],
                    'pc': np.zeros(1),
                }
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha_curr, alpha_other, r_rare, iTemp = params
        if self.fix_lr:
            alpha_other = alpha_curr
        if self.equal_reward and o == 0:
            o = -1
        self.wm['Q'], self.wm['pc'] = _MB_step_core_variables(alpha_curr, alpha_other, r_rare, self.p_transit, c, s, o, self.wm['Q'], self.wm['pc'])

    def _step_other_variables(self, params):
        alpha_curr, alpha_other, r_rare, iTemp = params
        if self.fix_lr:
            alpha_other = alpha_curr
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha_curr, alpha_other, r_rare, iTemp = params
        if self.fix_lr:
            alpha_other = alpha_curr
        outcomes = session['outcomes']
        if self.equal_reward:
             outcomes = outcomes * 2 - 1 # 0 -> -1, 1 -> 1
        DVs['trial_log_likelihood'], DVs['Q'], DVs['pc'], DVs['scores'], DVs['choice_probs'] = _MB_session_likelihood_core(
            alpha_curr, alpha_other, r_rare, iTemp, self.p_transit, session['choices'], session['second_steps'], outcomes,
            DVs['Q'], DVs['pc'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs