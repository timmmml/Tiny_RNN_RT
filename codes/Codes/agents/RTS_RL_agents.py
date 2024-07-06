import numpy as np
from numba import jit
from random import random, randint
import sys
from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

log_max_float = np.log(sys.float_info.max/2.1) # Log of largest possible floating point number. # 709.0407755486547


# Note: Numba is used to speed up likelihood evaluation. If numba is not available, 
# remove @jit decorators and the code will run, but with much slower likelihood evaluation.

# -------------------------------------------------------------------------------------
# Utility functions.
# -------------------------------------------------------------------------------------

@jit
def softmax(Q,T):
    "Softmax choice probs given values Q and inverse temp T."
    QT = Q * T
    QT[QT > log_max_float] = log_max_float # Protection against overflow in exponential.    
    expQT = np.exp(QT)
    return expQT/expQT.sum()

@jit
def array_softmax(Q,T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    # P = np.zeros(Q.shape)
    P = Q.copy()
    TdQ = -T*(Q[:,0]-Q[:,1])
    TdQ[TdQ > log_max_float] = log_max_float # Protection against overflow in exponential.    
    P[:,0] = 1./(1. + np.exp(TdQ))
    P[:,1] = 1. - P[:,0]
    return P

def array_softmax_square(Q,T, Q2_T):
    '''Array based calculation of softmax probabilities for binary choices.
    Q: Action values - array([n_trials,2])
    T: Inverse temp  - float.'''
    P = np.zeros(Q.shape)
    TdQ = -T*(Q[:,0]-Q[:,1])-Q2_T*(Q[:,0]-Q[:,1])**2
    TdQ[TdQ > log_max_float] = log_max_float # Protection against overflow in exponential.
    P[:,0] = 1./(1. + np.exp(TdQ))
    P[:,1] = 1. - P[:,0]
    return P

@jit
def protected_log(x):
    'Return log of x protected against giving -inf for very small values of x.'
    return np.log(((1e-200)/2)+(1-(1e-200))*x)

@jit
def protected_KL(choice_probs, real_choice_pr):
    # negative KL
    log_est_p = np.log(((1e-200) / 2) + (1 - (1e-200)) * choice_probs)
    nKL = (real_choice_pr * log_est_p).sum(-1)
    assert len(nKL)>10
    return nKL

def choose(P):
    "Takes vector of probabilities P summing to 1, returns integer s with prob P[s]"
    return sum(np.cumsum(P) < random())

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#
# Agents for reduced task version without choice at second step.
#
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
class BAS():
    ''' A Q1 (direct reinforcement) agent, in which the outcome directly increaces
    or decreaces the value of the action chosen at the first step.
    '''

    def __init__(self):
        self.name = 'Baseline'
        self.param_names = ['p']
        self.params = [0.5]
        self.param_ranges = ['unit']
        self.n_params = 1

    def session_likelihood(self, session, params, get_DVs=False):
        # Unpack trial events.
        choices, outcomes = (session.CTSO['choices'], session.CTSO['outcomes'])

        # Unpack parameters.
        p, = params

        # Variables.
        Q = np.zeros([session.n_trials + 1, 2])  # Model free action values.
        Q[:, 1] = protected_log(p)
        Q[:, 0] = protected_log(1 - p)

        # Evaluate choice probabilities and likelihood.
        choice_probs = array_softmax(Q, 1)
        assert np.isclose(choice_probs[0,1], p) and np.isclose(choice_probs[0,0], 1-p)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if get_DVs:
            return {'Q': Q, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood,
                    'scores': Q}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# Q1 agent
# -------------------------------------------------------------------------------------

class Q1():
    ''' A Q1 (direct reinforcement) agent, in which the outcome directly increaces
    or decreaces the value of the action chosen at the first step.
    '''

    def __init__(self):

        self.name = 'Q1'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate1trial(self, pre_state, inputs, params):
        pass

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td = np.zeros(2)
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c] = (1. - alpha) * Q_td[c] +  alpha * o    

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, outcomes = (session.CTSO['choices'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([session.n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen action.

            # update action values.
            Q_td[i+1, c] = (1. - alpha) * Q_td[i,c] +  alpha * o   
            Q_td[i+1,nc] = Q_td[i,nc]
            
        # Evaluate choice probabilities and likelihood. 
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if get_DVs:
            return {'Q_td':Q_td, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_td*iTemp}
        else:
            return session_log_likelihood


class Q1_prob(Q1):
    ''' A Q1 (direct reinforcement) agent, in which the outcome directly increaces
    or decreaces the value of the action chosen at the first step.
    '''

    def __init__(self):
        super().__init__()
        self.name = 'Q1 prob'


    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        alpha, iTemp = params

        # Variables.
        Q_td = np.zeros([session.n_trials + 1, 2])  # Model free action values.

        for i, (c, o) in enumerate(zip(choices, outcomes)):  # loop over trials.

            nc = 1 - c  # Not chosen action.

            # update action values.
            Q_td[i + 1, c] = (1. - alpha) * Q_td[i, c] + alpha * o
            Q_td[i + 1, nc] = Q_td[i, nc]

        # Evaluate choice probabilities and likelihood.
        choice_probs = array_softmax(Q_td, iTemp)
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)
        if get_DVs:
            return {'Q_td': Q_td, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_td*iTemp}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q1 perseveration agent.
# -------------------------------------------------------------------------------------

class Q1_prsv():
    ''' A Q1 agent with a perseveration bias.'''

    def __init__(self, params = None):

        self.name = 'Q1_prsv'
        self.param_names  = ['alpha', 'iTemp', 'prsv']
        self.params       = [ 0.5   ,  5.    ,  0.2 ]  
        self.param_ranges = ['unit' , 'pos'  , 'unc' ]
        self.n_params = 2
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, prsv = self.params  

        Q_td  = np.zeros(2) # TD action values excluding perseveration bias.
        Q_net = np.zeros(2) # Net action value including perseveration bias.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_net, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c] = (1. - alpha) * Q_td[c] +  alpha * o  
            Q_net[:] = Q_td[:]
            Q_net[c] += 1. * prsv

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Q0 agent
# -------------------------------------------------------------------------------------

class Q0():
    ''' A temporal difference agent without elegibility traces.'''

    def __init__(self):

        self.name = 'Q0'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate1trial(self, pre_state, inputs, params):
        pass

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2) # First  step action values. 
        Q_td_s = np.zeros(2) # Second step action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td_f, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * Q_td_s[s]   
            Q_td_s[s] = (1. - alpha) * Q_td_s[s] +  alpha * o     

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params

        #Variables.
        Q_td_f = np.zeros([session.n_trials + 1, 2])  # First  step action values.
        Q_td_s = np.zeros([session.n_trials + 1, 2])  # Second step action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen first step action.
            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] +  alpha * Q_td_s[i,s]  
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o    
            Q_td_f[i+1,nc] = Q_td_f[i,nc]
            Q_td_s[i+1,ns] = Q_td_s[i,ns]
            
        # Evaluate choice probabilities and likelihood.
        choice_probs = array_softmax(Q_td_f, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_td_f':Q_td_f, 'Q_td_s': Q_td_s, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_td_f*iTemp}
        else:
            return session_log_likelihood


class Q0_prob(Q0):
    ''' A temporal difference agent without elegibility traces.'''

    def __init__(self):
        super().__init__()
        self.name = 'Q0 prob'

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
            session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        alpha, iTemp = params

        # Variables.
        Q_td_f = np.zeros([session.n_trials + 1, 2])  # First  step action values.
        Q_td_s = np.zeros([session.n_trials + 1, 2])  # Second step action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            nc = 1 - c  # Not chosen first step action.
            ns = 1 - s  # Not reached second step state.

            # update action values.
            Q_td_f[i + 1, c] = (1. - alpha) * Q_td_f[i, c] + alpha * Q_td_s[i, s]
            Q_td_s[i + 1, s] = (1. - alpha) * Q_td_s[i, s] + alpha * o
            Q_td_f[i + 1, nc] = Q_td_f[i, nc]
            Q_td_s[i + 1, ns] = Q_td_s[i, ns]

        # Evaluate choice probabilities and likelihood.
        choice_probs = array_softmax(Q_td_f, iTemp)
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_td_f': Q_td_f, 'Q_td_s': Q_td_s, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_td_f*iTemp}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# TD lambda agent
# -------------------------------------------------------------------------------------

class Q_lambda():
    ''' A temporal difference agent with adjustable elegibility trace.
    '''

    def __init__(self, lambd = 0.5):

        self.name = 'Q_lambda'
        self.lambd = lambd
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2) # First  step action values. 
        Q_td_s = np.zeros(2) # Second step action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td_f, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * (Q_td_s[s] + self.lambd * (o - Q_td_s[s]))  
            Q_td_s[s] = (1. - alpha) * Q_td_s[s] +  alpha * o     

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  
        lambd = self.lambd

        #Variables.
        Q_td_f = np.zeros([session.n_trials + 1, 2])  # First  step action values.
        Q_td_s = np.zeros([session.n_trials + 1, 2])  # Second step action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            nc = 1 - c # Not chosen first step action.
            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_td_f[i+1,c] = (1. - alpha) * Q_td_f[i,c] +  alpha * (Q_td_s[i,s] + lambd * (o - Q_td_s[i,s]))
            Q_td_s[i+1,s] = (1. - alpha) * Q_td_s[i,s] +  alpha * o    
            Q_td_f[i+1,nc] = Q_td_f[i,nc]
            Q_td_s[i+1,ns] = Q_td_s[i,ns]
            
        # Evaluate choice probabilities and likelihood. 
        choice_probs = array_softmax(Q_td_f, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        return session_log_likelihood



# -------------------------------------------------------------------------------------
# Model based agent.
# -------------------------------------------------------------------------------------

class Model_based():
    ''' A model based agent which learns the values of the second step states 
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self, params = None, p_transit=0.8):
        self.name = 'Model based'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2
        if params:
            self.params = params
        self.p_transit = p_transit

    def simulate1trial(self, pre_state, inputs, params):
        assert len(pre_state.shape)==1
        alpha, iTemp = params
        Q_s    = pre_state.copy()
        c,s,o = inputs
        ns = 1 - s  # Not reached second step state.
        Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
        Q_mb   = self.p_transit * Q_s + (1-self.p_transit) * Q_s
        return Q_s

    def simulate(self, task, n_trials, get_DVs=False):

        alpha, iTemp = self.params  

        Q_s    = np.zeros([n_trials + 1, 2])  # Second_step state values.
        Q_mb   = np.zeros([n_trials + 1, 2])  # Model based action values.
        # Q_s  = np.zeros(2)  # Second_step state values.
        # Q_mb = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb[i], iTemp))
            s, o = task.trial(c)
            ns = 1 - s  # Not reached second step state.
            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

            # update action values.
            # Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_s[i+1, s] = (1. - alpha) * Q_s[i,s] +  alpha * o
            Q_s[i+1,ns] = Q_s[i,ns]
            Q_mb[i+1]   = self.p_transit * Q_s[i+1] + (1-self.p_transit) * Q_s[i+1,::-1]

        choice_probs = array_softmax(Q_mb, iTemp)
        if get_DVs:
            return choices, second_steps, outcomes, {'Q_s':Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs}
        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_s    = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb   = np.zeros([session.n_trials + 1, 2])  # Model based action values.


        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            ns = 1 - s # Not reached second step state.

            # update action values.
            Q_s[i+1, s] = (1. - alpha) * Q_s[i,s] +  alpha * o
            Q_s[i+1,ns] = Q_s[i,ns]
            
        # Evaluate choice probabilities and likelihood. 

        Q_mb = self.p_transit * Q_s + (1-self.p_transit) * Q_s[:,::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s':Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood


class Model_based_symm():
    ''' A model based agent which learns the values of the second step states
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self, params=None, p_transit=0.8, equal_reward=False):
        self.name = 'Model based symm'
        self.param_names = ['alpha', 'iTemp']
        self.params = [0.5, 5.]
        self.param_ranges = ['unit', 'pos']
        self.n_params = 2
        self.equal_reward = equal_reward
        if params:
            self.params = params
        self.p_transit = p_transit

    def simulate1trial(self, pre_state, inputs, params):
        assert len(pre_state.shape) == 1
        alpha, iTemp = params
        Q_s = pre_state.copy()
        c, s, o = inputs
        ns = 1 - s  # Not reached second step state.
        Q_s[s] = (1. - alpha) * Q_s[s] + alpha * o
        Q_s[ns] = (1. - alpha) * Q_s[ns] - alpha * o
        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s
        return Q_s

    def simulate(self, task, n_trials, get_DVs=False):
        assert NotImplementedError
        alpha, iTemp = self.params

        Q_s = np.zeros([n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([n_trials + 1, 2])  # Model based action values.
        # Q_s  = np.zeros(2)  # Second_step state values.
        # Q_mb = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb[i], iTemp))
            s, o = task.trial(c)
            ns = 1 - s  # Not reached second step state.
            choices[i], second_steps[i], outcomes[i] = (c, s, o)

            # update action values.
            # Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = (1. - alpha) * Q_s[i, ns] - alpha * o
            Q_mb[i + 1] = self.p_transit * Q_s[i + 1] + (1 - self.p_transit) * Q_s[i + 1, ::-1]

        choice_probs = array_softmax(Q_mb, iTemp)
        if get_DVs:
            return choices, second_steps, outcomes, {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs}
        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.
            if self.equal_reward and o == 0:
                o = -1
            ns = 1 - s  # Not reached second step state.

            # update action values.
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = (1. - alpha) * Q_s[i, ns] - alpha * o

        # Evaluate choice probabilities and likelihood.

        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood


class Model_based_mix():
    ''' A model based agent which learns the values of the second step states
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self, params=None, p_transit=0.8):
        self.name = 'Model based mix'
        self.param_names = ['alpha1','alpha2', 'iTemp']
        self.params = [0.5,0.5, 5.]
        self.param_ranges = ['unit','unit', 'pos']
        self.n_params = 3
        if params:
            self.params = params
        self.p_transit = p_transit

    def simulate1trial(self, pre_state, inputs, params):
        assert len(pre_state.shape) == 1
        alpha1, alpha2, iTemp = params
        Q_s = pre_state.copy()
        c, s, o = inputs
        ns = 1 - s  # Not reached second step state.
        alpha = alpha1 if o==1 else alpha2
        Q_s[s] = (1. - alpha) * Q_s[s] + alpha * o
        if o == 0: Q_s[ns] = (1. - alpha) * Q_s[ns] - alpha * o
        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s
        return Q_s

    def simulate(self, task, n_trials, get_DVs=False):

        alpha1, alpha2, iTemp = self.params

        Q_s = np.zeros([n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([n_trials + 1, 2])  # Model based action values.
        # Q_s  = np.zeros(2)  # Second_step state values.
        # Q_mb = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb[i], iTemp))
            s, o = task.trial(c)
            ns = 1 - s  # Not reached second step state.
            choices[i], second_steps[i], outcomes[i] = (c, s, o)

            # update action values.
            # Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            alpha = alpha1 if o == 1 else alpha2
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = Q_s[i, ns]
            if o == 0: Q_s[i + 1, ns] = (1. - alpha) * Q_s[i, ns] - alpha * o
            Q_mb[i + 1] = self.p_transit * Q_s[i + 1] + (1 - self.p_transit) * Q_s[i + 1, ::-1]

        choice_probs = array_softmax(Q_mb, iTemp)
        if get_DVs:
            return choices, second_steps, outcomes, {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs}
        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha1, alpha2, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            alpha = alpha1 if o == 1 else alpha2
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = Q_s[i, ns]
            if o == 0: Q_s[i + 1, ns] = (1. - alpha) * Q_s[i, ns] - alpha * o
        # Evaluate choice probabilities and likelihood.

        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood


class Model_based_mix_decay():
    ''' A model based agent which learns the values of the second step states
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self, params=None, p_transit=0.8):
        self.name = 'Model based mix decay'
        self.param_names = ['alpha1','alpha2', 'b', 'iTemp']
        self.params = [0.4, 0.2, 0.4, 5.]
        self.param_ranges = ['unit','unit','unit', 'pos']
        self.n_params = 4
        if params:
            self.params = params
        self.p_transit = p_transit

    def simulate1trial(self, pre_state, inputs, params):
        assert len(pre_state.shape) == 1
        assert NotImplementedError
        alpha1, alpha2, b, iTemp = params
        Q_s = pre_state.copy()
        c, s, o = inputs
        ns = 1 - s  # Not reached second step state.

        if o == 1:
            Q_s[s] = alpha1 * Q_s[s] + 1
            Q_s[ns] = Q_s[ns] - b
        else:
            Q_s[s] = Q_s[s] + alpha2 * (Q_s[ns] - Q_s[s])
            Q_s[ns] = Q_s[ns]


        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s
        return Q_s

    def simulate(self, task, n_trials, get_DVs=False):
        assert NotImplementedError
        alpha1, alpha2, iTemp = self.params

        Q_s = np.zeros([n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([n_trials + 1, 2])  # Model based action values.
        # Q_s  = np.zeros(2)  # Second_step state values.
        # Q_mb = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb[i], iTemp))
            s, o = task.trial(c)
            ns = 1 - s  # Not reached second step state.
            choices[i], second_steps[i], outcomes[i] = (c, s, o)

            # update action values.
            # Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            alpha = alpha1 if o == 1 else alpha2
            Q_s[i + 1, s] = alpha * Q_s[i, s] + o
            Q_s[i + 1, ns] = Q_s[i, ns]
            if o == 0: Q_s[i + 1, ns] = alpha * Q_s[i, ns]
            Q_mb[i + 1] = self.p_transit * Q_s[i + 1] + (1 - self.p_transit) * Q_s[i + 1, ::-1]

        choice_probs = array_softmax(Q_mb, iTemp)
        if get_DVs:
            return choices, second_steps, outcomes, {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs}
        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha1, alpha2, b, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            if o == 1:
                Q_s[i + 1, s] = alpha1 * Q_s[i, s] + 1
                Q_s[i + 1, ns] = Q_s[i, ns] - b
            else:
                Q_s[i + 1, s] = Q_s[i, s] + alpha2 * (Q_s[i, ns] - Q_s[i, s])
                Q_s[i + 1, ns] = Q_s[i, ns]

        # Evaluate choice probabilities and likelihood.

        Q_mb = self.p_transit * Q_s + (1 - self.p_transit) * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood

class CPB_model_free():
    def __init__(self):
        self.name = 'Change point model free'
        self.param_names  = ['iTemp']
        self.params       = [5.    ]
        self.param_ranges = ['pos'  ]
        self.n_params = 1

    def simulate1trial(self, pre_state, inputs, params):
        pass

    #@jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])
        assert second_steps[0] in [0.1, 0.2, 0.4] # hack the second step to be the hazard rate
        # Unpack parameters.
        beta = 1 - second_steps[0] # decay rate
        iTemp, = params

        # Variables.
        r_mean = 0.5
        Q = np.ones([session.n_trials + 1, 2]) * r_mean

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            nc = 1 - c  # Not chosen action.
            # update action values.
            Q[i + 1, c] = beta * o + (1-beta) * r_mean
            Q[i + 1, nc] = beta * Q[i, nc] + (1-beta) * r_mean

        # Evaluate choice probabilities and likelihood.

        choice_probs = array_softmax(Q, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q': Q, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q*iTemp}
        else:
            return session_log_likelihood


class CPB_model_based():
    def __init__(self):
        self.name = 'Change point model based'
        self.param_names  = ['gamma', 'iTemp']
        self.params       = [0.1, 5.    ]
        self.param_ranges = ['unit','pos'  ]
        self.n_params = 2

        table_path = r'D:\OneDrive\Documents\git_repo\explore-exploit-old'
        self.Q_tables = {
            0.1: np.load(f'{table_path}\\Q_tables_H01.npy', allow_pickle=True).item(),
            0.2: np.load(f'{table_path}\\Q_tables_H02.npy', allow_pickle=True).item(),
            0.4: np.load(f'{table_path}\\Q_tables_H04.npy', allow_pickle=True).item(),
        }

    #@jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])
        assert second_steps[0] in [0.1, 0.2, 0.4] # hack the second step to be the hazard rate
        # Unpack parameters.
        hazard_rate = second_steps[0]
        beta = 1 - hazard_rate # decay rate
        gamma, iTemp = params

        # Variables.
        r_mean = 0.5
        Q = np.ones([session.n_trials + 1, 2]) * r_mean
        assert np.max(outcomes) <= 1
        rewards_max_100 = (outcomes * 100).astype(np.int32)
        Q_policies = self._eval_policy(choices, rewards_max_100, gamma, hazard_rate, reward=99) / 100
        Q[1:, :] = Q_policies
        choice_probs = array_softmax(Q, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q': Q, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q*iTemp}
        else:
            return session_log_likelihood

    def _eval_policy(self, actions, rewards, gamma, hazard_rate, reward=99):
        """evaluate the actions of subject with off policy simulation
        action, reward. 1d nparray

        Return: Q_policies: [n_trials, 2]
        """
        Q_table = self.Q_tables[hazard_rate][str(round(gamma, 2))]

        n_trials = actions.shape[0]
        Q_policies = np.zeros([n_trials, 2])

        NoSS = 0
        r_knw = int((1 + reward) / 2) -1
        r_unk = int((1 + reward) / 2) -1
        for i in range(n_trials):
            action = actions[i]

            if i > 0:
                shift = (actions[i] != actions[i-1]).astype(np.int32)
            else:
                shift = 1

            if shift == 1:  # the agent shift/explore
                Q_policies[i, action] = Q_table[r_unk, r_knw, NoSS, 1]  # Q_stay
                Q_policies[i, 1 - action] = Q_table[r_unk, r_knw, NoSS, 0]  # Q_shift
                NoSS = 0  # this is the index, corresponds to n = 1
                if i > 0:  # if shift, R_unk is the previous R_knw
                    r_unk = rewards[i-1] -1
            else:  # the agent stay/exploit
                Q_policies[i, action] = Q_table[r_unk, r_knw, NoSS, 0]  # Q_shift
                Q_policies[i, 1 - action] = Q_table[r_unk, r_knw, NoSS, 1]  # Q_stay
                NoSS += 1
            r_knw = rewards[i] -1

        return Q_policies


class Model_based_decay():
    def __init__(self, p_transit=0.8):
        self.name = 'Model based decay'
        self.param_names  = ['alpha', 'beta', 'iTemp']
        self.params       = [ 0.5   , 0.5   , 5.    ]
        self.param_ranges = ['unit' , 'unit' ,'pos'  ]
        self.n_params = 3
        self.p_transit = p_transit

    def simulate1trial(self, pre_state, inputs, params):
        pass

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, beta, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = beta * Q_s[i, ns]
            #Q_s[i + 1, ns] = Q_s[i, ns]

        # Evaluate choice probabilities and likelihood.

        Q_mb = self.p_transit * Q_s + (1-self.p_transit) * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood

class Model_based_prob(Model_based):
    ''' A model based agent which learns the values of the second step states
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self):
        super().__init__()
        self.name = 'Model based prob'

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
            session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        alpha, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = Q_s[i, ns]

        # Evaluate choice probabilities and likelihood.

        Q_mb = 0.8 * Q_s + 0.2 * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood


class Model_based_prob_decay(Model_based_prob):
    ''' A model based agent which learns the values of the second step states
    through reward prediction errors and then calculates the values of the first step
    actions using the transition probabilties.
    '''

    def __init__(self):
        super().__init__()
        self.param_names  = ['alpha', 'beta', 'iTemp']
        self.params       = [ 0.5   , 0.5   , 5.    ]
        self.param_ranges = ['unit' , 'unit' ,'pos'  ]
        self.n_params = 3
        self.name = 'Model based prob decay'

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
            session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        alpha, beta, iTemp = params

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            # Q_s[i + 1, :] *= beta
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = beta * Q_s[i, ns]

        # Evaluate choice probabilities and likelihood.

        Q_mb = 0.8 * Q_s + 0.2 * Q_s[:, ::-1]
        choice_probs = array_softmax(Q_mb, iTemp)
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q_mb*iTemp}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# Model based perseveration agent.
# -------------------------------------------------------------------------------------

class MB_prsv():
    ''' A model based agent with perseveration bias'''

    def __init__(self, params = None):
        self.name = 'MB_prsv'
        self.param_names  = ['alpha', 'iTemp', 'prsv']
        self.params       = [ 0.5   ,  5.    ,  0.2  ]  
        self.param_ranges = ['unit' , 'pos'  , 'unc' ]
        self.n_params = 3
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, prsv = self.params  

        Q_s   = np.zeros(2)  # Second_step state values.
        Q_mb  = np.zeros(2)  # Model based action values.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_mb   = 0.8 * Q_s + 0.2 * Q_s[::-1]
            Q_mb[c] += 1. * prsv # Perseveration bias.

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Model based agent with transition matrix leraning.
# -------------------------------------------------------------------------------------

class MB_trans_learn():
    ''' Model based agent which learns the transtion matrix from experience.
    '''

    def __init__(self, params = None):
        self.name = 'MB trans. learn.'
        self.param_names  = ['alpha', 'iTemp', 'tlr' ]
        self.params       = [ 0.5   ,  5.    ,  0.5  ]  
        self.param_ranges = ['unit' , 'pos'  , 'unit']
        self.n_params = 3
        if params:
            self.params = params

    def simulate(self, task, n_trials):

        alpha, iTemp, tlr = self.params  

        Q_s  = np.zeros(2)      # Second_step state values.
        Q_mb = np.zeros(2)      # Model based action values.
        tp   = np.ones(2) * 0.5 # Transition probabilities for first step actions, coded as 
                                # probability of reaching second step state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s, o = task.trial(c)
            # update action values and transition probabilities.
            Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            tp[c]  = (1. - tlr) * tp[c] + tlr * (s == 0)
            Q_mb   = tp * Q_s[0] + (1 - tp) * Q_s[1]

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Reward_as_cue agent
# -------------------------------------------------------------------------------------

class Reward_as_cue():
    '''Agent which uses reward location as a cue for state.  The agent learns seperate values
    for actions following distinct outcomes and second_steps on the previous trial.
    '''

    def __init__(self):

        self.name = 'Reward as cue'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.05   ,  6.   ] 
        self.param_ranges = ['unit' , 'pos'  ] 
        self.n_params = 2

    def simulate1trial(self, pre_state, inputs, params):
        pass

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td = np.zeros([2, 2, 2]) # Indicies: action, prev. second step., prev outcome
        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_td[:,ps,po], iTemp)) 
            s, o = task.trial(c)
            # update action values.
            Q_td[c,ps,po] = (1. - alpha) * Q_td[c,ps,po] +  alpha * o  
            ps, po = s, o  

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_td = np.zeros([2, 2, 2]) # Indicies: action, prev. second step., prev outcome
        Q    = np.zeros([session.n_trials+1, 2])  # Active action values.

        # ps = randint(0,1) # previous second step.
        # po = randint(0,1) # previous outcome.
        ps = po = 0 # deterministic starting point.
        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            Q[i,0] = Q_td[0,ps,po]
            Q[i,1] = Q_td[1,ps,po]

            # update action values.
            Q_td[c,ps,po] = (1. - alpha) * Q_td[c,ps,po] +  alpha * o  
            ps, po = s, o  
        assert i == session.n_trials - 1
        Q[session.n_trials,0] = Q_td[0,ps,po]
        Q[session.n_trials,1] = Q_td[1,ps,po]
        # Evaluate choice probabilities and likelihood. 

        choice_probs = array_softmax(Q, iTemp)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q': Q, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q*iTemp}
        else:
            return session_log_likelihood


class Reward_as_cue_prob(Reward_as_cue):
    def __init__(self):
        super().__init__()
        self.name = 'Reward as cue prob'

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
            session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])
        # Unpack parameters.
        alpha, iTemp = params

        #Variables.
        Q_td = np.zeros([2, 2, 2]) # Indicies: action, prev. second step., prev outcome
        Q    = np.zeros([session.n_trials, 2])  # Active action values.

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            Q[i,0] = Q_td[0,ps,po]
            Q[i,1] = Q_td[1,ps,po]

            # update action values.
            Q_td[c,ps,po] = (1. - alpha) * Q_td[c,ps,po] +  alpha * o
            ps, po = s, o

        # Evaluate choice probabilities and likelihood.

        choice_probs = array_softmax(Q, iTemp)
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_td':Q_td, 'Q': Q, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': Q*iTemp}
        else:
            return session_log_likelihood

# -------------------------------------------------------------------------------------
# Reward_as_cue_fixed agent
# -------------------------------------------------------------------------------------

class RAC_fixed():
    '''Agent which deterministically follows the following decison rules
    mapping second step and outcome onto next choice (s, o --> c)  .   
    0, 1 --> 0
    0, 0 --> 1
    1, 1 --> 1
    1, 0 --> 0
    '''

    def __init__(self):

        self.name = 'RAC fixed.'
        self.param_names  = []
        self.params       = []  
        self.n_params = 0

    def simulate(self, task, n_trials):

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = int(ps == po)
            s, o = task.trial(c)
            ps, po = s, o  

            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes

# -------------------------------------------------------------------------------------
# Latent state agent
# -------------------------------------------------------------------------------------

class Latent_state():
    
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses 
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self, good_prob=0.8):

        self.name = 'Latent state'
        self.param_names  = ['p_r' , 'p_lapse']
        self.params       = [ 0.1  , 0.1      ]
        self.param_ranges = ['half', 'half'   ]   
        self.n_params = 2
        self.good_prob = good_prob

    def simulate(self, task, n_trials):

        p_r, p_lapse = self.params

        p_1 = 0.5 # Probability world is in state 1.

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):

            # Generate trial events.
            c = (p_1 > 0.5) == (random() > p_lapse)
            s, o = task.trial(c)
            # Bayesian update of state probabilties given observed outcome.
            p_1 = p_o_1[s,o] * p_1 / (p_o_1[s,o] * p_1 + p_o_0[s,o] * (1 - p_1))   
            # Update of state probabilities due to possibility of block reversal.
            p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)  


            choices[i], second_steps[i], outcomes[i]  = (c, s, o)

        return choices, second_steps, outcomes


    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, p_lapse = params

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        p_1    = np.zeros(session.n_trials + 1) # Probability world is in state 1.
        p_1[0] = 0.5 

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)): # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i+1] = p_o_1[s,o] * p_1[i] / (p_o_1[s,o] * p_1[i] + p_o_0[s,o] * (1 - p_1[i]))   
            # Update of state probabilities due to possibility of block reversal.
            p_1[i+1] = (1 - p_r) * p_1[i+1] + p_r * (1 - p_1[i+1])  

        # Evaluate choice probabilities and likelihood. 
        choice_probs = np.zeros([session.n_trials + 1, 2])
        choice_probs[:,1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        choice_probs[:,0] = 1 - choice_probs[:,1] 
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'p_1':p_1, 'choice_probs': choice_probs}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# Latent state agent
# -------------------------------------------------------------------------------------

class Latent_state_softmax():
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self, good_prob=0.8, params=None):

        self.name = 'Latent state softmax'
        self.param_names = ['p_r', 'iTemp']
        self.params = [0.1, 5.]
        self.param_ranges = ['half', 'pos']
        self.n_params = 2
        self.good_prob = good_prob
        if params:
            self.params = params

    def simulate1trial(self, pre_state, inputs, params):
        assert len(pre_state.shape)==1
        p_r, iTemp = params
        p_0, p_1    = pre_state
        p_1 = min(max(p_1, 0),1)
        p_0 = min(max(p_0, 0),1)

        c,s,o = inputs
        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

        def get_new_p1(p_1):
            # Bayesian update of state probabilties given observed outcome.
            p_1 = p_o_1[s, o] * p_1 / (p_o_1[s, o] * p_1 + p_o_0[s, o] * (1 - p_1))
            # Update of state probabilities due to possibility of block reversal.
            p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)
            return p_1
        return np.array([1-get_new_p1(1-p_0),get_new_p1(p_1)])

    def simulate(self, task, n_trials, get_DVs=False):

        p_r, iTemp = self.params

        p_1 = np.zeros(n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5
        # p_1 = 0.5  # Probability world is in state 1.

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(np.array([1 - p_1[i], p_1[i]]), iTemp))
            s, o = task.trial(c)
            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

            choices[i], second_steps[i], outcomes[i] = (c, s, o)

        choice_probs = np.zeros([n_trials + 1, 2])
        choice_probs[:, 1] = p_1
        choice_probs[:, 0] = 1 - p_1
        choice_probs = array_softmax(choice_probs, iTemp)
        if get_DVs:
            return choices, second_steps, outcomes, {'p_1': p_1, 'choice_probs': choice_probs}
        return choices, second_steps, outcomes

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, iTemp = params

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
        p_1 = np.zeros(session.n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

            # Evaluate choice probabilities and likelihood.
        scores = np.zeros([session.n_trials + 1, 2])
        scores[:, 1] = p_1
        scores[:, 0] = 1 - p_1
        scores *= iTemp
        choice_probs = array_softmax(scores, 1)
        # choice_probs[:, 1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        # choice_probs[:, 0] = 1 - choice_probs[:, 1]
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'p_1': p_1, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': scores}
        else:
            return session_log_likelihood


class Latent_state_softmax_bias():
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self, good_prob=0.8):

        self.name = 'Latent state softmax biased'
        self.good_prob = good_prob
        self.param_names = ['p_r', 'bias1', 'bias2','iTemp']
        self.params = [0.1, 1, 1, 5.]
        self.param_ranges = ['half', 'pos','pos', 'pos']
        self.n_params = 4

    def simulate1trial(self, pre_state, inputs, params):
        pass

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, bias1, bias2, iTemp = params

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
        p_o_0[0] *= bias1
        p_o_0[1] *= bias2
        p_1 = np.zeros(session.n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5


        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

            # Evaluate choice probabilities and likelihood.
        scores = np.zeros([session.n_trials + 1, 2])
        scores[:, 1] = p_1
        scores[:, 0] = 1 - p_1
        scores *= iTemp
        choice_probs = array_softmax(scores, 1)
        # choice_probs[:, 1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        # choice_probs[:, 0] = 1 - choice_probs[:, 1]
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'p_1': p_1, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': scores}
        else:
            return session_log_likelihood


class Latent_state_softmax_prob(Latent_state_softmax):
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self, good_prob=0.8):
        super().__init__(good_prob=good_prob)
        self.name = 'Latent state softmax prob'

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        p_r, iTemp = params

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

        p_1 = np.zeros(session.n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

            # Evaluate choice probabilities and likelihood.
        scores = np.zeros([session.n_trials + 1, 2])
        scores[:, 1] = p_1
        scores[:, 0] = 1 - p_1
        scores *= iTemp
        choice_probs = array_softmax(scores, 1)
        # choice_probs[:, 1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        # choice_probs[:, 0] = 1 - choice_probs[:, 1]
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'p_1': p_1, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': scores}
        else:
            return session_log_likelihood

class Latent_state_softmax_prob_bias(Latent_state_softmax):
    '''Agent which belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses
    the action which leads to the best second step in that state with probability (1- p_lapse)
    '''

    def __init__(self, good_prob=0.8):
        super().__init__(good_prob=good_prob)
        self.name = 'Latent state softmax prob biased'
        self.param_names = ['p_r', 'bias1', 'bias2','iTemp']
        self.params = [0.1, 1, 1, 5.]
        self.param_ranges = ['half', 'pos','pos', 'pos']
        self.n_params = 4

    @jit
    def session_likelihood(self, session, params, get_DVs=False):

        # Unpack trial events.
        choices, second_steps, outcomes, real_choice_pr = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'], session.CTSO['choice_pr'])

        # Unpack parameters.
        p_r, bias1, bias2, iTemp = params

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
        p_o_0[0] *= bias1
        p_o_0[1] *= bias2
        p_1 = np.zeros(session.n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

            # Evaluate choice probabilities and likelihood.
        scores = np.zeros([session.n_trials + 1, 2])
        scores[:, 1] = p_1
        scores[:, 0] = 1 - p_1
        scores *= iTemp
        choice_probs = array_softmax(scores, 1)
        # choice_probs[:, 1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        # choice_probs[:, 0] = 1 - choice_probs[:, 1]
        trial_log_likelihood = protected_KL(choice_probs[:session.n_trials], real_choice_pr)
        # trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'p_1': p_1, 'choice_probs': choice_probs, 'session_log_likelihood': session_log_likelihood, 'scores': scores}
        else:
            return session_log_likelihood



class MB0LS0():
    '''
    '''
    def __init__(self, good_prob=0.8, params=None):
        self.name = 'Model based latent state'
        self.param_names = ['alpha', 'iTemp_mb','p_r', 'iTemp_ls','w']
        self.params = [0.5, 5., 0.1, 5., 0.5]
        self.param_ranges = ['unit', 'pos','half', 'pos', 'unit']
        self.n_params = 5
        self.good_prob = good_prob
        if params:
            self.params = params

    def simulate(self, task, n_trials, get_DVs=False, varying_w=None):
        alpha, iTemp_mb, p_r, iTemp_ls, w_mb = self.params
        w_mb = None
        if varying_w is not None:
            assert len(varying_w) == n_trials+1
        Q_s = np.zeros([n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([n_trials + 1, 2])  # Model based action values.
        Q_ls = np.zeros([n_trials + 1, 2]) # LS action values.
        Q_mix = np.zeros([n_trials + 1, 2])  # Model based + LS action values.

        p_1 = np.zeros(n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5
        # p_1 = 0.5  # Probability world is in state 1.
        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]
        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.

        choices, second_steps, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mix[i], 1))
            s, o = task.trial(c)
            ns = 1 - s  # Not reached second step state.
            choices[i], second_steps[i], outcomes[i] = (c, s, o)

            # update action values.
            # Q_s[s] = (1. - alpha) * Q_s[s] +  alpha * o
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = Q_s[i, ns]
            Q_mb[i + 1] = 0.8 * Q_s[i + 1] + 0.2 * Q_s[i + 1, ::-1]

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])
            Q_ls[i + 1] = np.array([1 - p_1[i + 1], p_1[i + 1]])
            if varying_w is not None:
                w_mb = varying_w[i + 1]
            Q_mix[i + 1] = w_mb * Q_mb[i + 1] * iTemp_mb + (1 - w_mb) * Q_ls[i + 1] * iTemp_ls

        choice_probs = array_softmax(Q_mix, 1)
        Q_mb *= iTemp_mb
        Q_ls *= iTemp_ls
        if get_DVs:
            return choices, second_steps, outcomes, {'Q_s': Q_s, 'Q_mb': Q_mb, 'p_1': p_1, 'Q_ls': Q_ls, 'Q_mix': Q_mix, 'choice_probs': choice_probs, 'varying_w': varying_w}
        return choices, second_steps, outcomes

    # @jit
    def session_likelihood(self, session, params=None, get_DVs=False, varying_w=None):

        # Unpack trial events.
        choices, second_steps, outcomes = (
        session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['outcomes'])

        # Unpack parameters.
        if params is None: params = self.params
        alpha, iTemp_mb, p_r, iTemp_ls, w_mb = params
        if varying_w is not None:
            w_mb = varying_w[..., None]

        # Variables.
        Q_s = np.zeros([session.n_trials + 1, 2])  # Second_step state values.
        Q_mb = np.zeros([session.n_trials + 1, 2])  # Model based action values.

        p_o_1 = np.array([[self.good_prob    , 1 - self.good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - self.good_prob, self.good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1  # Probability of observed outcome given world in state 0.
        p_1 = np.zeros(session.n_trials + 1)  # Probability world is in state 1.
        p_1[0] = 0.5

        for i, (c, s, o) in enumerate(zip(choices, second_steps, outcomes)):  # loop over trials.

            ns = 1 - s  # Not reached second step state.

            # update action values.
            Q_s[i + 1, s] = (1. - alpha) * Q_s[i, s] + alpha * o
            Q_s[i + 1, ns] = Q_s[i, ns]

            # Bayesian update of state probabilties given observed outcome.
            p_1[i + 1] = p_o_1[s, o] * p_1[i] / (p_o_1[s, o] * p_1[i] + p_o_0[s, o] * (1 - p_1[i]))
            # Update of state probabilities due to possibility of block reversal.
            p_1[i + 1] = (1 - p_r) * p_1[i + 1] + p_r * (1 - p_1[i + 1])

        # Evaluate choice probabilities and likelihood.

        Q_mb = 0.8 * Q_s + 0.2 * Q_s[:, ::-1]
        Q_ls = np.zeros([session.n_trials + 1, 2])
        Q_ls[:, 1] = p_1
        Q_ls[:, 0] = 1 - p_1

        Q_mb *= iTemp_mb
        Q_ls *= iTemp_ls
        Q_mix = w_mb * Q_mb + (1 - w_mb) * Q_ls

        choice_probs = array_softmax(Q_mix, 1)
        trial_log_likelihood = protected_log(choice_probs[np.arange(session.n_trials), choices])
        session_log_likelihood = np.sum(trial_log_likelihood)

        if get_DVs:
            return {'Q_s': Q_s, 'Q_mb': Q_mb, 'p_1': p_1, 'Q_ls': Q_ls, 'Q_mix': Q_mix, 'choice_probs': choice_probs, 'varying_w': varying_w}
        else:
            return session_log_likelihood


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
#
# Agents for original (Daw et al 2011) task version.
#
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Q1 agent - original task version.
# -------------------------------------------------------------------------------------

class Q1_orig():
    ''' A direct reinforcement (Q1) agent for use with the version of the task used in 
    Daw et al 2011 with choices at the second step.
    '''

    def __init__(self):

        self.name = 'Q1, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f, iTemp))      # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * o    
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * o    
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q0 agent - original task version.
# -------------------------------------------------------------------------------------

class Q0_orig():
    ''' Q(0) agent for use with the version of the task used in Daw et al 2011.'''

    def __init__(self):

        self.name = 'Q0, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f, iTemp))      # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * np.max(Q_td_s[s,:])  
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.

        Q_td_f = np.zeros(2)     # First step action values.
        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c] = (1. - alpha) * Q_td_f[c] +  alpha * np.max(Q_td_s[s,:])  
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Model based agent - original task version.
# -------------------------------------------------------------------------------------

class Model_based_orig():
    ''' A model based agent for use with the version of the task used in 
    Daw et al 2011 with choices at the second step.
    '''

    def __init__(self):
        self.name = 'Model based, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_s = np.zeros([2,2])  # Second_step action values, indicies: [state, action]
        Q_mb   = np.zeros(2)      # Model based action values.

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c = choose(softmax(Q_mb, iTemp)) 
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.  

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o
            Q_s = np.max(Q_td_s, 1) # State values are max action value available in each state.
            Q_mb   = 0.7 * Q_s + 0.3 * Q_s[::-1]

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp = params  

        #Variables.
        Q_mb = np.zeros(2)        # Model based action values.
        Q_td_s = np.zeros([2,2])  # Second_step action values, indicies: [state, action]

        Q_mb_array   = np.zeros([session.n_trials + 1, 2])     # Model based action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_mb_array[i,j] = Q_mb[j]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o
            Q_s_0 = np.max(Q_td_s[0,:]) # Max action value in second step state 0.
            Q_s_1 = np.max(Q_td_s[1,:]) # Max action value in second step state 1.
            Q_mb[0] = 0.7 * Q_s_0 + 0.3 * Q_s_1
            Q_mb[1] = 0.3 * Q_s_0 + 0.7 * Q_s_1  

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_mb_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Reward_as_cue agent - original task version.
# -------------------------------------------------------------------------------------

class Reward_as_cue_orig():
    '''Reward as cue agent for original version of task.
    '''

    def __init__(self):

        self.name = 'Reward as cue, original task.'
        self.param_names  = ['alpha', 'iTemp', 'alpha_s', 'iTemp_s']
        self.params       = [ 0.05  ,   10.  ,  0.5     ,     5.   ] 
        self.param_ranges = ['unit' , 'pos'  , 'unit'   ,   'pos'  ] 
        self.n_params = 4

    def simulate(self, task, n_trials):

        alpha, iTemp, alpha_s, iTemp_s = self.params  

        Q_td_f = np.zeros([2, 2, 2]) # First step action values, indicies: action, prev. second step., prev outcome
        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c   = choose(softmax(Q_td_f[:,ps,po], iTemp)) # First step action.
            s   = task.first_step(c)                    # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp_s))   # Second step action.
            o   = task.second_step(s, c_s)              # Trial outcome.   

            # update action values.
            Q_td_f[c,ps,po] = (1. - alpha) * Q_td_f[c,ps,po] +  alpha * o  
            ps, po = s, o  
            Q_td_s[s,c_s] = (1. - alpha_s) * Q_td_s[s,c_s] + alpha_s * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes

    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        alpha, iTemp, alpha_s, iTemp_s = params  

        #Variables.
        Q_td_f = np.zeros([2, 2, 2]) # First step action values, indicies: action, prev. second step., prev outcome
        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        Q_td_f_array = np.zeros([session.n_trials + 1, 2])     # First step action values.
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.

        ps = 0 # previous second step.
        po = 0 # previous outcome.

        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Store action values in array, note: inner loop over actions is used rather than 
            # slice indexing because slice indexing is not currently supported by numba.

            for j in range(2):
                Q_td_f_array[i,j] = Q_td_f[j, ps, po]
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # update action values.
            Q_td_f[c,ps,po] = (1. - alpha) * Q_td_f[c,ps,po] +  alpha * o  
            ps, po = s, o  
            Q_td_s[s,c_s] = (1. - alpha_s) * Q_td_s[s,c_s] + alpha_s * o

        # Evaluate choice probabilities and likelihood. 
        choice_probs_f = array_softmax(Q_td_f_array, iTemp)
        choice_probs_s = array_softmax(Q_td_s_array, iTemp_s)
        trial_log_likelihood_f = protected_log(choice_probs_f[np.arange(session.n_trials), choices  ])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood


# -------------------------------------------------------------------------------------
# Reward_as_cue_fixed agent
# -------------------------------------------------------------------------------------

class RAC_fixed_orig():
    '''RAC_fixed agent for original task version, using deterministic strategy at first
    step then TD at second step.
    '''

    def __init__(self):

        self.name = 'RAC fixed, original task'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        task.reset(n_trials)

        ps = randint(0,1) # previous second step.
        po = randint(0,1) # previous outcome.

        for i in range(n_trials):
            # Generate trial events.
            c = int(ps == po)
            s   = task.first_step(c)                    # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp))   # Second step action.
            o   = task.second_step(s, c_s)              # Trial outcome.   
            ps, po = s, o  

            # update action values.
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


# -------------------------------------------------------------------------------------
# Latent state agent - original task version.
# -------------------------------------------------------------------------------------

class Latent_state_orig():
    
    '''Version of latent state agent for original version of task.

    Agent belives that there are two states of the world:

    State 0, Second step 0 reward prob = good_prob, sec. step 1 reward prob = 1 - good_prob
    State 1, Second step 1 reward prob = good_prob, sec. step 0 reward prob = 1 - good_prob

    Agent believes the probability that the state of the world changes on each step is p_r.

    The agent infers which state of the world it is most likely to be in, and then chooses 
    the action which leads to the best second step in that state with probability (1- p_lapse)

    Choices at the second step are mediated by action values learnt through TD.
    '''

    def __init__(self):

        self.name = 'Latent state, original task.'
        self.param_names  = ['p_r' , 'p_lapse', 'alpha', 'iTemp']
        self.params       = [ 0.1  , 0.1      ,  0.5   ,    6.  ]
        self.param_ranges = ['half', 'half'   , 'unit' , 'pos'  ]   
        self.n_params = 4

    def simulate(self, task, n_trials):

        p_r, p_lapse, alpha, iTemp = self.params
        good_prob = 0.625

        p_1 = 0.5 # Probability world is in state 1.

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):

            # Generate trial events.
            c   = (p_1 > 0.5) == (random() > p_lapse) # First step choice.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # Bayesian update of state probabilties given observed outcome.
            p_1 = p_o_1[s,o] * p_1 / (p_o_1[s,o] * p_1 + p_o_0[s,o] * (1 - p_1))   
            # Update of state probabilities due to possibility of block reversal.
            p_1 = (1 - p_r) * p_1 + p_r * (1 - p_1)  

            # Update second step action values.

            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


    @jit
    def session_likelihood(self, session, params):

        # Unpack trial events.
        choices, second_steps, choices_s, outcomes = (session.CTSO['choices'], session.CTSO['second_steps'], session.CTSO['choices_s'], session.CTSO['outcomes'])

        # Unpack parameters.
        p_r, p_lapse, alpha, iTemp = params
        good_prob = 0.625

        p_o_1 = np.array([[good_prob    , 1 - good_prob],   # Probability of observed outcome given world in state 1.
                          [1 - good_prob, good_prob    ]])  # Indicies:  p_o_1[second_step, outcome]

        p_o_0 = 1 - p_o_1               # Probability of observed outcome given world in state 0.

        p_1    = np.zeros(session.n_trials + 1) # Probability world is in state 1.
        p_1[0] = 0.5 

        Q_td_s = np.zeros([2,2])     # Second step action values, indicies: [state, action]
        Q_td_s_array = np.zeros([session.n_trials + 1, 2])     # Second step action values for second step reached on each trial.


        for i, (c, s, c_s, o) in enumerate(zip(choices, second_steps, choices_s, outcomes)): # loop over trials.

            # Bayesian update of state probabilties given observed outcome.
            p_1[i+1] = p_o_1[s,o] * p_1[i] / (p_o_1[s,o] * p_1[i] + p_o_0[s,o] * (1 - p_1[i]))   
            # Update of state probabilities due to possibility of block reversal.
            p_1[i+1] = (1 - p_r) * p_1[i+1] + p_r * (1 - p_1[i+1])  

            for j in range(2): # Store second step action values in array
                Q_td_s_array[i,j] = Q_td_s[s,j]

            # Update second step action values.

            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o


        # Evaluate choice probabilities and likelihood. 
        choice_probs = np.zeros([session.n_trials + 1, 2]) # First step choice probabilities.
        choice_probs[:,1] = (p_1 > 0.5) * (1 - p_lapse) + (p_1 <= 0.5) * p_lapse
        choice_probs[:,0] = 1 - choice_probs[:,1] 
        choice_probs_s = array_softmax(Q_td_s_array, iTemp) # Second step choice probabilities.
        trial_log_likelihood_f = protected_log(choice_probs[np.arange(session.n_trials), choices])
        trial_log_likelihood_s = protected_log(choice_probs_s[np.arange(session.n_trials), choices_s])
        session_log_likelihood = np.sum(trial_log_likelihood_f) + np.sum(trial_log_likelihood_s)  
        return session_log_likelihood

# -------------------------------------------------------------------------------------
# Q0 agent - original task version.
# -------------------------------------------------------------------------------------

class Random_first_step_orig():
    ''' Agent which makes random choice at first step.'''

    def __init__(self):

        self.name = 'Rand FS, original task.'
        self.param_names  = ['alpha', 'iTemp']
        self.params       = [ 0.5   ,  5.    ]  
        self.param_ranges = ['unit' , 'pos'  ]
        self.n_params = 2

    def simulate(self, task, n_trials):

        alpha, iTemp = self.params  

        Q_td_s = np.zeros([2,2]) # Second step action values, indicies: [state, action]

        choices, second_steps, choices_s, outcomes = (np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int), np.zeros(n_trials, int))

        task.reset(n_trials)

        for i in range(n_trials):
            # Generate trial events.
            c   = randint(0,1)                        # First step action.
            s   = task.first_step(c)                  # Second step state.
            c_s = choose(softmax(Q_td_s[s,:], iTemp)) # Second_step action.
            o   = task.second_step(s, c_s)            # Trial outcome.   

            # update action values. 
            Q_td_s[s,c_s] = (1. - alpha) * Q_td_s[s,c_s] + alpha * o

            choices[i], second_steps[i], choices_s[i], outcomes[i]  = (c, s, c_s, o)

        return choices, second_steps, choices_s, outcomes


