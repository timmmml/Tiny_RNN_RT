class BaseTrainer(object):
    """Interface to train a specific agent."""
    def __init__(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError