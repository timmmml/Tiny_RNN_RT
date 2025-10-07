import numpy as np
from numba import jit
from random import random, randint
from .core import TwoStepModelCoreCSO, _compute_loglik, _step_other_variables

@jit(nopython=True)
def _MB_step_core_variables(alpha, alpha_pers, r_rare, r_pers, p_transit, c, s, o, Q, Q_a, Q_pers):
    nc = 1 - c  # Not chosen first step action.
    ns = 1 - s  # Not reached second step state.

    Q_a_new = Q_a.copy()
    Q_pers_new = Q_pers.copy()
    # update action values.
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
    Q_a_new[c] = (1. - alpha) * Q_a[c] + alpha * u
    Q_a_new[nc] = (1. - alpha) * Q_a[nc] - alpha * u
    if (c == s and o == 0) or (c != s and o == 1):
        Q_pers_new[c] = (1. - alpha_pers) * Q_pers[c] + alpha_pers * r_pers
        Q_pers_new[nc] = (1. - alpha_pers) * Q_pers[nc] - alpha_pers * r_pers
    Q_new = Q_a_new + Q_pers_new
    return Q_new, Q_a_new, Q_pers


@jit(nopython=True)
def _MB_session_likelihood_core(alpha, alpha_pers, r_rare, r_pers, iTemp, p_transit, choices, second_steps, outcomes, Q, Q_a, Q_pers, scores, choice_probs, n_trials):
    trial_log_likelihood = np.zeros(n_trials)
    for trial in range(n_trials):
        c, s, o = choices[trial], second_steps[trial], outcomes[trial]
        trial_log_likelihood[trial] = _compute_loglik(choice_probs[trial], c)
        Q[trial + 1], Q_a[trial + 1], Q_pers[trial + 1] = _MB_step_core_variables(alpha, alpha_pers, r_rare, r_pers, p_transit, c, s, o, Q[trial],Q_a[trial], Q_pers[trial])
        scores[trial + 1], choice_probs[trial + 1] = _step_other_variables(iTemp, Q[trial + 1])
    return trial_log_likelihood, Q, Q_a, Q_pers, scores, choice_probs

class Model_based_symm_acthist(TwoStepModelCoreCSO):
    def __init__(self, p_transit=0.8):
        super().__init__()
        self.name = 'Model based symm action history'
        self.param_names = ['alpha', 'alpha_pers', 'r_rare', 'r_pers', 'iTemp']
        self.params = [0.5, 0.5, 0.5, 0.5, 5.]
        self.param_ranges = ['unit', 'unit', 'unit', 'unit', 'pos']
        self.n_params = 5
        self.p_transit = p_transit
        self.state_vars = ['Q_a', 'Q_pers']

    def _init_core_variables(self, wm, params):
        if wm is None:
            self.wm = {
                'Q': np.zeros(2),
                'Q_a': np.zeros(2),
                'Q_pers': np.zeros(2),
            }
        else:
            if 'h0' in wm:
                raise NotImplementedError
            else:
                self.wm = wm

    def _step_core_variables(self, trial_event, params):
        (c, s, o) = trial_event
        alpha, alpha_pers, r_rare, r_pers, iTemp = params
        self.wm['Q'], self.wm['pc'] = _MB_step_core_variables(alpha, alpha_pers, r_rare, r_pers, self.p_transit, c, s, o, self.wm['Q'], self.wm['Q_a'], self.wm['Q_pers'])

    def _step_other_variables(self, params):
        alpha, alpha_pers, r_rare, r_pers, iTemp = params
        self.wm['scores'], self.wm['choice_probs'] = _step_other_variables(iTemp, self.wm['Q'])

    def _session_likelihood_core(self, session, params, DVs):
        alpha, alpha_pers, r_rare, r_pers, iTemp = params
        outcomes = session['outcomes']
        DVs['trial_log_likelihood'], DVs['Q'], DVs['Q_a'], DVs['Q_pers'], DVs['scores'], DVs['choice_probs'] = _MB_session_likelihood_core(
            alpha, alpha_pers, r_rare, r_pers, iTemp, self.p_transit, session['choices'], session['second_steps'], outcomes,
            DVs['Q'], DVs['Q_a'], DVs['Q_pers'], DVs['scores'], DVs['choice_probs'], session['n_trials'])
        return DVs