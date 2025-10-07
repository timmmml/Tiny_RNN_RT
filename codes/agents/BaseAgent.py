import os
from copy import deepcopy
import json
import joblib
import numpy as np
from path_settings import *


class BaseAgent(object):
    """The base class for all agents (RNN or cog agents).

    Attributes:
        model: the agent's model.
        config: the path to save/load the agent's model.
    """

    def __init__(self):
        """Initialize the agent."""
        if not hasattr(self, 'config'):
            self.config = None
        return

    def forward(self, *args, **kwargs):
        """The agent takes a few trials (batched) and return outputs and internal agent states.
        """
        raise NotImplementedError

    def load(self, model_path):
        """The agent's parameters are loaded from file.
        """
        raise NotImplementedError

    def save(self, model_path):
        """The agent's parameters are saved to file.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Typical usage:
            output_dict = agent(input)
        """
        return self.forward(*args, **kwargs)

    def simulate(self, task, config, save=True):
        """The agent takes and interacts with a task instance.
        """
        raise NotImplementedError


    def save_config(self):
        """Save config to disk."""
        model_path = self.config['model_path']
        os.makedirs(MODEL_SAVE_PATH / model_path, exist_ok=True)
        joblib.dump(self.config, MODEL_SAVE_PATH / model_path / 'config.pkl')

        config_json = deepcopy(self.config)
        try:
            for k in config_json.keys():
                # print(k, type(config_json[k]))
                if 'index' in k:
                    if not isinstance(config_json[k], list):
                        config_json[k] = config_json[k].tolist()
                if 'path' in k:
                    config_json[k] = str(config_json[k])
                if isinstance(config_json[k], np.int32):
                    config_json[k] = int(config_json[k])
            with open(MODEL_SAVE_PATH / model_path / 'config_easyread.json', 'w') as f:
                json.dump(config_json, f, indent=4)
        except:
            print('config saving failed')
            for k in config_json.keys():
                print(k, type(config_json[k]))
            raise
