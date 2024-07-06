"""This module contains the Simulation class for the project.

The dataset class is responsible for simulating data for the project.
- Task 0: revesral learning task (Li et al. Task A)
- Task 1: two-stage task (Li et al. Task B)
- Task 2: transition-reversal two-stage task (Li et al. Task C)
- Task 3: two-arm bandit
- Task 4: pulse-accumulation
"""
import numpy as np
import pyddm

class Task():
    def __init__(self, task_id, agent, trial_number):
        self.task_id = task_id
        self.task_string = None
        if task_id == 0:
            self.task_string = 'reversal learning task (Li et al. Task A)'
        elif task_id == 1:
            self.task_string = 'two-stage task (Li et al. Task B)'
        elif task_id == 2:
            self.task_string = 'transition-reversal two-stage task (Li et al. Task C)'
        elif task_id == 3:
            self.task_string = 'two-arm bandit'
            raise ValueError('task id not implemented yet')
        elif task_id == 4:
            self.task_string = 'pulse-accumulation'
            raise ValueError('task id not implemented yet')
        else:
            raise ValueError('task_id not recognized')
        self.trial_number = trial_number
        self.stims = []
        self.choices = []
        self.agent = None
        self.RT = []

    def simulate(self):
        """Generates stim and reward for the task."""
        pass


class ReverseLearning(Task):
    """Deterministic A-S relationships, S-R relationship reverse with contingency"""
    def __init__(self, agent, trial_number, LOW=0.2, reverse_prob=0.2):
        super().__init__(0, agent, trial_number)
        self.S2 = 0  # second stage state that is associated with high reward.
        self.LOW = LOW  # low reward probability
        self.HIGH = 1 - self.LOW  # high reward probability
        self.low = 0
        self.high = 1 - self.low
        self.reverse_prob = reverse_prob  # expect 5 trials to learn and then reverse.
        self.rlow = 0
        self.rhigh = 1

    def simulate(self):
        """Simulates the task for the agent."""
        for _ in range(self.trial_number):
            self.__simulate_one_trial()

    def __simulate_one_trial(self):
        """Simulates one trial for the agent."""
        a = self.choices[-1]
        s = a if np.random.rand() < self.high else 1 - a  # state is the action in this task
        if np.random.rand() < self.reverse_prob:
            self.S2 = 1 - self.S2
        if s == self.S2:
            if np.random.rand() < self.LOW:
                r = self.rlow
            else:
                r = self.rhigh
        else:
            if np.random.rand() < self.HIGH:
                r = self.rhigh
            else:
                r = self.rlow
        self.stims.append((a, s, r))
        self.agent.update(self.stims[-1])
        self.choices.append(self.agent.choose())
        self.RT.append(self.agent.rt)

class Simulation():
    def __init__(self, task, episodes, episode_length, mod1_size, mod2_size):
        self.task = task
        self.episodes = episodes
        self.episode_length = episode_length
        self.mod1_size = mod1_size
        self.mod2_size = mod2_size

    def __call__(self):
        pass
