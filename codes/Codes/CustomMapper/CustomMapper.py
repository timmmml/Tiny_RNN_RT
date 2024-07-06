"""This module remaps the dataset to my purposes (said Tim).
Target: one-package solution for mapping from JI-AN's code structure to mine.

Detail:
from: CogModel (DVs, model_config)
to: Trainer, RNN_Model (implemented my way, for training on my laptop), DataLoaders
goal: call one function to start training.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from pathlib import Path
from Network_models import Trainer as t
from Network_models import RNN_models as r
import joblib
import pyddm
import scipy

class CustomMapper:
    def __init__(self, configs):
        self.configs = configs

        self.cog_model = None
        self.load_cog_model(configs["cog_model"]) # Load the cog model with some flexibility

        self.trainer = None
        self.initialise_trainer(configs)

        self.data = None
        self.remap_data(configs)

        self.dataloaders = None
        self.initialise_dataloaders(configs)

    # NOTE: the below subroutines would include config as argument, for the convenience of re-doing them.
    def load_cog_model(self, cog_model):
        if type(cog_model) == str:
            cog_model_path = Path(cog_model)
            self.cog_model = joblib.load(cog_model_path)
            return
        self.cog_model = cog_model # If it is directly provided

    def initialise_trainer(self, configs):
        self.trainer = t.Trainer(configs)

    def add_RT(self, config):
        """Add RT to the data. Add some pyddm params for this model"""
        # Let's use one model for now
        if config.get("DDM_model", None) is None:
            self.rt_model = pyddm.gddm(
                drift = lambda t, strength_A, strength_B, driftscale, bias: (strength_A - strength_B) * driftscale + bias,
                nondecision = lambda T, ndt_s, ndt_mu: scipy.stats.norm(ndt_mu, ndt_s).pdf(T),
                T_dur = 10,
                parameters = {"driftscale": self.driftscale, "bias": self.bias, "ndt_s": self.ndt_s, "ndt_mu": self.ndt_mu},
                conditions = ["strength_A", "strength_B"],
            )
        else:
            self.rt_model = config["DDM_model"]

    def remap_data(self, config):
        """Based on the model configuration id, remap the decision variables into training material

        """
        task_id = config["training_config"]["task_id"]
        match task_id:
            case "1.1":
                self._map_to_task_1_1()
            case "1.2":
                self._map_to_task_1_2()
            case _:
                raise NotImplementedError

    def initialise_dataloaders(self, configs):
        pass

    def _map_to_task_1_1(self):
        inputs = self.cog_models_DV


class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.data = features
        self.labels = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



