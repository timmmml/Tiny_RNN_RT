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
import joblib
import pyddm
import scipy
from numba import jit
import pandas as pd
from collections import defaultdict

class CustomMapper:
    def __init__(self, configs):
        self.configs = configs

        self.cog_model = None
        if "cog_model" in configs:
            self.load_cog_model(configs["cog_model"]) # Load the cog model with some flexibility
        else:
            self.cog_model = None

        self.cog_data = None
        self.load_cog_data(configs["cog_data"])

        self.trainer = None
        self.initialise_trainer(configs)

        self.redo_count = 0
        self.rt_model = None
        self.add_RT(configs)

        self._cog_data_frame()

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

    def load_cog_data(self, cog_data):
        if type(cog_data) == str:
            cog_data_path = Path(cog_data)
            self.cog_data = joblib.load(cog_data_path)
            return
        elif type(cog_data) == type(Path("")):
            self.cog_data = joblib.load(cog_data)
            return
        self.cog_data = cog_data # If it is directly provided

    def initialise_trainer(self, configs = None):
        if configs is None:
            configs = self.configs
        self.trainer = t.Trainer(configs)

    def reload_RT_model(self, config = None):
        self.rt_model = pyddm.gddm(
            drift=lambda t, strength_A, strength_B, driftscale, bias: (
                                                                                  strength_A - strength_B) * driftscale + bias,
            nondecision=lambda T, ndt_s, ndt_mu: scipy.stats.norm(ndt_mu, ndt_s).pdf(T),
            T_dur=10,
            parameters={"driftscale": self.driftscale, "bias": self.bias, "ndt_s": self.ndt_s,
                        "ndt_mu": self.ndt_mu},
            conditions=["strength_A", "strength_B"],
        )

    def add_RT(self, config = None):
        """Add RT to the data. Add some pyddm params for this model"""
        # Let's use one model for now

        if config is None:
            config = self.configs
        self.dt = config.get("dt", 0.02)  # detault to 20ms as implemented with Jaffe et al.
        self.T = config.get("T", 100)  # default to 100 timepoints (2 seconds)
        cutoff = self.T * self.dt
        if config.get("DDM_model", None) is None:
            self.driftscale = config.get("driftscale", 10)
            self.bias = config.get("bias", 0)
            self.ndt_s = config.get("ndt_s", 0.1)
            self.ndt_mu = config.get("ndt_mu", 0.3)
            self.rt_model = pyddm.gddm(
                drift=lambda t, strength_A, strength_B, driftscale, bias: (strength_A - strength_B) * driftscale + bias,
                nondecision=lambda T, ndt_s, ndt_mu: scipy.stats.norm(ndt_mu, ndt_s).pdf(T),
                T_dur=10,
                parameters={"driftscale": self.driftscale, "bias": self.bias, "ndt_s": self.ndt_s, "ndt_mu": self.ndt_mu},
                conditions=["strength_A", "strength_B"],
            )
        else:
            self.rt_model = config["DDM_model"]

        for i in range(len(self.cog_data["score"])):
            if len(self.cog_data["score"][i]) != 100:
                self.cog_data["score"][i] = self.cog_data["score"][i][:-1, :]

        score_blocks = self.cog_data["score"]

        action_blocks = self.cog_data["action"]
        self.cog_data['RTs'] = [[0 for _ in range(len(score_blocks[0]))] for _ in range(len(score_blocks))]
        for i in range(len(score_blocks)):
            print("Block", i)
            score_block = score_blocks[i]
            for j in range(len(score_block)):
                solution = self.rt_model.solve(
                    {"strength_A": score_block[j, 1], "strength_B": score_block[j, 0]}).sample(k = 1)
                # Mapping index 0 (in scores) to Lower, 1 to Higher
                rts = [solution.choice_lower, solution.choice_upper]
                while solution.undecided or (config['redo_choices'] == False and not len(rts[action_blocks[i][j]])):
                    self.redo_count += 1
                    solution = self.rt_model.solve(
                        {"strength_A": score_block[j, 1], "strength_B": score_block[j, 0]}).sample(k = 1)
                    rts = [solution.choice_lower, solution.choice_upper]
                self.cog_data["RTs"][i][j] = min(rts[action_blocks[i][j]][0], cutoff)


    def remap_data(self, config = None):
        """Based on the model configuration id, remap the decision variables into training material

        """
        if config is None:
            config = self.configs
        task_id = config["training_config"]["task_id"]
        match task_id:
            case "1.1":
                self._map_to_task_1_1()
            case "1.2":
                self._map_to_task_1_2()
            case "Task-DyVA":
                self._map_to_task_DyVA(config)
            case _:
                raise NotImplementedError

    def initialise_dataloaders(self, config = None):
        # to feed to the trainer for training!
        if config is None:
            config = self.configs
        train_dataset = CustomDataset(self.input_data[:int(0.8 * len(self.input_data)), ...], self.choice_data[:int(0.8 * len(self.choice_data)), ...])
        val_dataset = CustomDataset(self.input_data[int(0.8 * len(self.input_data)):, ...], self.choice_data[int(0.8 * len(self.choice_data)):, ...])
        self.dataloaders = {
            "train": DataLoader(train_dataset, batch_size= config["model_specs"]['model_params']["batch_size"], shuffle=True),
            "val": DataLoader(val_dataset, batch_size= config["model_specs"]["model_params"]["batch_size"], shuffle=True)
        }


    def _map_to_task_1_1(self):
        raise NotImplementedError

    def _map_to_task_1_2(self):
        raise NotImplementedError

    def _map_to_task_DyVA(self, config):
        if config.get("task", None) != "PRL_Bartolo":
            raise NotImplementedError("Task not implemented")

        self.input_noise = config.get("input_noise", 0.1)
        # We know that there are several dimensions in the input. Namely, a past action, a past reward, and a past second-stage state.
        # Let's add a go-cue as well. This is a binary variable that indicates the start of a trial.
        print("At this stage, I implement one task and the simplified configuration only")
        # Note: in this configuration we ignore the block setting (which can be problematic).

        choice_data = np.zeros((len(self.cog_data["block"].unique()), (len(self.cog_data["score"][self.cog_data['block'] == 0]) - 1) * self.T, 2))
        print(choice_data.shape)
        input_data = np.zeros((len(self.cog_data['block'].unique()), (len(self.cog_data["score"][self.cog_data['block'] == 0]) - 1) * self.T, config['model_specs']['model_params']['u_dim']))
        for i in range(len(self.cog_data["block"].unique())):
            if config['model_specs']['model_params']['u_dim'] == 2:
                # PRL without second stages and simplified
                input_data[i, :, 0] = np.array(self.trial_to_batch(mode = "past action", block = i))
                input_data[i, :, 1] = np.array(self.trial_to_batch(mode = "reward", block = i))
            if config['model_specs']['model_params']['u_dim'] == 3:
                # models with second stages
                input_data[i, :, 0] = np.array(self.trial_to_batch(mode = "past action", block = i))
                input_data[i, :, 1] = np.array(self.trial_to_batch(mode = "reward", block = i))
                input_data[i, :, 2] = np.array(self.trial_to_batch(mode = "stage2", block = i))
            if config['model_specs']['model_params']['u_dim'] == 4:
                # PRL with two block types

                input_data[i, :, 0] = np.array(self.trial_to_batch(mode = "past action_dim1", block = i))
                input_data[i, :, 1] = np.array(self.trial_to_batch(mode = "past action_dim2", block = i))
                input_data[i, :, 2] = np.array(self.trial_to_batch(mode = "reward", block = i))
                input_data[i, :, 3] = np.array(self.trial_to_batch(mode = "coherence", block = i))
            choice_data[i, :, 0] = np.array(self.trial_to_batch(mode = "current action L", block = i))
            choice_data[i, :, 1] = np.array(self.trial_to_batch(mode = "current action H", block = i))

        self.input_data = input_data
        self.choice_data = choice_data



    def trial_to_batch(self, mode, block = 0):
        block_data = self.cog_data[self.cog_data['block'] == block].reset_index()
        returned = []
        match mode:
            case "past action":
                # past action is treated as the input.
                # modelling starts with the second timepoint.
                for trial in range(1, len(block_data)):
                    returned.extend(list(np.array([block_data["action"][trial - 1]] * self.T) + np.random.randn(self.T) * self.input_noise))
                return returned
            case "reward":
                for trial in range(1, len(block_data)):
                    returned.extend(list(np.array([block_data["reward"][trial - 1]] * self.T) + np.random.randn(self.T) * self.input_noise))
                return returned
            case "current action L":
                # Map L to 0
                for trial in range(1, len(block_data)):
                    rt_bin = int(block_data["RTs"][trial] // self.dt)
                    trial_batch = [0] * self.T
                    if 0 <= rt_bin < self.T:
                        trial_batch[rt_bin] = 1 - block_data["action"][trial]
                    returned.extend(trial_batch)
                return returned
            case "current action H":
                # Map H to 1
                for trial in range(1, len(block_data)):
                    rt_bin = int(block_data["RTs"][trial] // self.dt)
                    trial_batch = [0] * self.T
                    if 0 <= rt_bin < self.T:
                        trial_batch[rt_bin] = block_data["action"][trial]
                    returned.extend(trial_batch)
                return returned

            case "stage2":
                raise NotImplementedError

    def _cog_data_frame(self):
        if type(self.cog_data) == pd.DataFrame:
            return
        columns = [
            "action",
            "stage2",
            "reward",
            "score",
            "RTs"
        ]
        cog_data_overall = dict(self.cog_data)
        cog_data = defaultdict(list)

        for i in range(len(cog_data_overall[columns[0]])):
            cog_data['block'].extend([i] * len(cog_data_overall[columns[0]][i]))
            for key in columns:
                cog_data[key].extend(cog_data_overall[key][i])
        columns.append("block")
        self.cog_data = pd.DataFrame(cog_data, columns=columns)
        self.cog_data['RTs'] = self.cog_data['RTs'].apply(lambda x: min(x, self.T * self.dt))


class CustomDataset(Dataset):
    def __init__(self, u, x):
        self.x = x
        self.u = u

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.u[idx], self.x[idx]



