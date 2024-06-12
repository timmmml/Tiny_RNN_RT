"""This module contains the Simulation class for the project.

The dataset class is responsible for simulating data for the project.
- Task 0: simple RL task to benchmark
- Task 1: two-arm bandit
- Task 2: pulse-accumulation
"""
class Simulation():
    def __init__(self, task, episodes, episode_length, mod1_size, mod2_size):
        self.task = task
        self.episodes = episodes
        self.episode_length = episode_length
        self.mod1_size = mod1_size
        self.mod2_size = mod2_size

    def __call__(self):
        pass
