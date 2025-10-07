"""Trainer.py

This module provides training wrappers and loss functions for various agent and RNN models in the Tiny_RNN_RT project.
It supports flexible training loops, checkpointing, validation, and logging for a variety of architectures, including standard RNNs, slow-fast RNNs, discretized RNNs, and dynamic agent models.

Key features:
- Modular Trainer base class for model/optimizer/scheduler management and checkpointing.
- Specialized trainers for different model types (RTRNN, SlowFast, Discretised, DynAgent, TaskDyVA).
- Loss functions for joint RT and choice prediction, slow-fast architectures, and dynamic agents.
- Integrated TensorBoard logging and early stopping. (note that I stopped using TensorBoard after devving RTRNN, so SlowFast and Discretised may be buggy)
- Designed for compatibility with PyTorch DataLoader.

Classes:
    - Trainer: Base class for model training, checkpointing, and optimizer management.
    - RTRNNTrainer: Trainer for RNNs predicting both RT and choice.
    - SlowFastTrainer: Trainer for two-timescale (slow/fast) RNNs.
    - DiscretisedTrainer: Trainer for discretized RNNs with categorical outputs.

    - DynAgentTrainer: Trainer for dynamic agent models. (not important)
    - TaskDyVATrainer: Trainer for DyVA models with ELBO loss. (not used; variational Bayes was crazy slow and hard to tune hyperparams to avoid NaNs)

Losses:
    - RTLoss: Computes both RT and choice loss for joint prediction models.
    - RTSlowFastLoss: Loss for slow-fast RNNs, combining RT and choice losses.

    - DynAgentLoss: Compound loss for dynamic agent models, including regularization. (for DynAgent)
    - ELBO: Evidence Lower Bound loss for variational models. (for TaskDyVA)

Typical usage:
    - Instantiate a trainer with a config dict specifying model, optimizer, and training parameters.
    - Call `train_loop` or `train` with data loaders or simulation parameters.
    - Use built-in checkpointing, validation, and logging for robust training.
    - Note

See the project notebooks (e.g., sim2test.ipynb, RNN_RT_discretised.ipynb) for example usage and integration with data pipelines.
"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from path_settings import *
import importlib
from copy import deepcopy

class Trainer():
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        self.config = config
        self.save_path = config.get("save_path", MODEL_SAVE_PATH / "default.pth")
        self.check_path = config.get("check_path", MODEL_SAVE_PATH / "checkpoints")
        self.log_path = config.get("log_path", LOG_PATH / "default")
        self.model_specs = config.get("model_specs", {})
        self.model = self._model_type(self.model_specs)
        self.device = config.get("device", "cpu")
        self.optimizer_specs = config.get("optimizer_specs", {})
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        self.lr_init = self.optimizer_specs["optimizer_params"]["lr"]

        self.loss_fn = None
        self.model = self.model.to(self.device)

        if config.get("scheduler", False):
            self._scheduler()

    def _scheduler(self):
        module = __import__("torch.optim.lr_scheduler", fromlist=[self.config["scheduler"]["name"]])
        self.scheduler = getattr(module, self.config["scheduler"]["name"])(self.optimizer,
                                                                      **self.config["scheduler"]["params"])
    def refresh(self):
        reinitialise_weights(self.model)
        self.model.out = None
        self.optimizer = self._optimizer_type(self.optimizer_specs)
        if self.config.get("scheduler", False):
            self.scheduler = self._scheduler()
        self.model.to(self.device)

    def save_checkpoint(self, path, e, loss):
        torch.save(
            {
                "epoch": e,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(self.device)  # Just in case it's not there already

    def save_model(self, path, full=False):
        if full:  # Save the full model
            torch.save(self.model, path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_best_model(self):
        self.model.load_state_dict(self.best_model)

    def load_model(self, path, full=False):
        if full:
            self.model = torch.load(path)
        else:
            self.model.load_state_dict(torch.load(path))

        self.model.to(self.device)  # Just in case it's not there already

    def forward(self, features, labels):
        self.model.eval()
        with torch.no_grad():
            output = self.model(features)
            loss = self.loss_fn(output[:, -self.action_period:, :], self.model.pred, labels)
        return output, loss

    def _model_type(self, model_specs):
        model_name = model_specs["model_name"]
        model = None
        try:
            # Dynamically import the module
            module = __import__(f"{model_specs['model_path']}", fromlist=[model_name])
            # Get the model class
            ModelClass = getattr(module, model_name)
            # Instantiate the model with parameters
            model = ModelClass(model_specs["model_params"])
        except Exception as e:
            print(f"Failed to load model: {e}")
        return model

    def _optimizer_type(self, optimizer_specs):
        optimizer_name = optimizer_specs["optimizer_name"]
        optimizer = None

        try:
            # Get the optimizer class from the optim module
            OptimizerClass = getattr(optim, optimizer_name)
            # Instantiate the optimizer with the model parameters and provided optimizer parameters
            optimizer = OptimizerClass(self.model.parameters(), **optimizer_specs["optimizer_params"])
        except AttributeError:
            print(f"Optimizer {optimizer_name} not found in torch.optim")
        except Exception as e:
            print(f"Failed to initialize optimizer: {e}")

        return optimizer
    
def reinitialise_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            reinitialise_weights(layer)


class RTRNNTrainer(Trainer): 
    def __init__(self, config=None):
        super(RTRNNTrainer, self).__init__(config)
        self.rt_dist_specs = config.get("rt_dist_specs", {})
        self.loss_fn = RTLoss(self.rt_dist_specs)
        self.rt_weight = config.get("rt_weight", 0.5)

    def train(
            self, train_loader, val_loader, epochs=10, save_interval=10, reset_interval=200, save_path=None,
            check_path=None, log_path=None, early_stop=100):

        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        self.model = self.model.to(self.device)

        for e in range(epochs):
            self.model.train()
            for i, (u, x) in enumerate(train_loader):
                u = u.to(self.device).float()
                x = x.to(self.device).float()
                self.optimizer.zero_grad()
                output = self.model(u)
                rt_loss, choice_loss = self.loss_fn(output, x)
                loss = self.rt_weight * rt_loss + choice_loss
                
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss.item(), i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training rt loss",
                                       rt_loss.item(),
                                       i + e * len(train_loader))
                self.writer.add_scalar("training choice loss",
                                       choice_loss.item(),
                                       i + e * len(train_loader))

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(str(check_path) + "/checkpoint" + str(e) + ".pth", e, loss)

            if (e + 1) % reset_interval == 0 and e / epochs < 0.75:
                if self.scheduler is not None:
                    self._scheduler()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_init

            if self.validate(val_loader, e, early_stop):
                break
        self.save_model(save_path)

    def validate(self, val_loader, epoch, early_stop):
        self.model.eval()
        val_total_loss = 0
        val_choice_loss = 0 
        val_rt_loss = 0
        with torch.no_grad():
            for i, (u, x) in enumerate(val_loader):
                u = u.to(self.device).float()
                x = x.to(self.device).float()
                output = self.model(u)
                rt_loss, choice_loss = self.loss_fn(output, x)
                loss = self.rt_weight * rt_loss + choice_loss

                val_total_loss += loss
                val_choice_loss += choice_loss
                val_rt_loss += rt_loss
            avg_val_loss = val_total_loss / len(val_loader)
            avg_val_choice_loss = val_choice_loss / len(val_loader)
            avg_val_rt_loss = val_rt_loss / len(val_loader)
 
            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)               

            self.writer.add_scalar("validation loss", loss.item(), epoch)

            # Log additional loss components
            self.writer.add_scalar("validation rt loss",
                                    avg_val_rt_loss.item(),
                                    epoch)
            self.writer.add_scalar("validation choice loss",
                                    avg_val_choice_loss.item(),
                                    epoch)

            if avg_val_loss < self.best_val_loss_split:
                self.best_val_loss_split = avg_val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= early_stop:
                    print("Early stopping")
                    return (1)
            print(avg_val_loss)
            return (0)



class SlowFastTrainer(Trainer):
    """
    Thin wrapper around your previous Trainer that:
        • expects a RTSlowFastRNN model
        • uses RTSlowFastLoss
    """

    def __init__(self,
                config: dict = None):
        super(SlowFastTrainer, self).__init__(config)

        self.SFloss_specs = config.get("SFloss_specs", {})
        self.loss_fn    = RTSlowFastLoss(self.SFloss_specs)
        self.model.to(self.device)

    # ---------------------------------------------------------------------
    def _step(self, batch, is_train=True, stage = 1):
        """one optimisation / validation step"""
        u, x = batch            # unpack dataloader tuple

        
        u = u.to(self.device).float()
        x = x.to(self.device).float()
        y_true = x[..., 0].to(self.device).float().unsqueeze(-1)
        rt_true= x[..., 1].to(self.device).float().unsqueeze(-1)

        if is_train: self.optimizer.zero_grad()

        out = self.model(u, slow_only = 1 - stage)
        total, choice_loss, rt_loss = self.loss_fn(out, (y_true, rt_true), stage=stage)

        if is_train:
            total.backward()

        nan_grad = False
        for name, p in self.model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN gradient found in {name}")
                print(f"Gradient: {p.grad}")
                nan_grad = True
        if nan_grad:
            raise RuntimeError("NaN gradients found")
        self.optimizer.step()

        if stage == 0 : 
            return total.item(), choice_loss.item(), rt_loss
        else:
            return total.item(), choice_loss.item(), rt_loss.item()

    # ---------------------------------------------------------------------
    def train_loop(self,
                   train_loader,
                   val_loader,
                   stage = 1,  # 0 is pretraining
                   epochs=20,
                   early_stop=100, log_path=None, 
                   device = None, 
                   ):

        if log_path is None:
            log_path = self.log_path
        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")
        self.best_val = float("inf")
        if device is not None: 
            self.model.to(device)
            # self.optimizer.to(device)
            old_device = self.device
            self.device = device
            
                                              
        for ep in range(epochs):
            # -------- training ----------
            self.model.train()
            for i,batch in enumerate(train_loader):
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)          # move exp_avg etc.
                tot, ch, rt = self._step(batch, is_train=True, stage=stage)
                global_step = ep*len(train_loader)+i
                self.writer.add_scalar("train/total",  tot, global_step)
                self.writer.add_scalar("train/choice", ch , global_step)
                self.writer.add_scalar("train/rt"    , rt , global_step)
                print(f"Epoch {ep} - Step {i} - train loss: {tot:.4f} | choice loss: {ch:.4f} | rt loss: {rt:.4f}")

            # -------- validation --------
            self.model.eval()
            with torch.no_grad():
                totals = []
                for batch in val_loader:
                    tot, ch, rt = self._step(batch, is_train=False, stage = stage)
                    totals.append(tot)
                val_mean = sum(totals)/len(totals)
                self.writer.add_scalar("val/total", val_mean, ep)

            # lr scheduler / early stop
            if self.scheduler: self.scheduler.step(val_mean)

            self.save_checkpoint(str(self.check_path) + f"/checkpoint{ep}.pth", ep, val_mean)
            if val_mean < self.best_val:
                self.best_val = val_mean
                self.epochs_no_improve = 0
                self.save_checkpoint(str(self.check_path) + f"/best_model.pth", ep, val_mean)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= early_stop:
                    print(f"Early stopping @epoch {ep}")
                    break
            print(f"Epoch {ep} - val loss: {val_mean:.4f} | best: {self.best_val:.4f}")
            self.device = old_device if device is not None else self.device  # reset device if changed
            self.model.to(self.device)  # ensure model is on the correct device
            # self.optimizer.to(self.device)  # ensure optimizer is on the correct device

class DiscretisedTrainer(Trainer):
    """
    Thin wrapper around your previous Trainer that:
        • expects a RTSlowFastRNN model
        • uses RTSlowFastLoss
    """

    def __init__(self,
                config: dict = None):
        super(DiscretisedTrainer, self).__init__(config)

        self.loss_type = config.get("loss_type", "mse").lower()
        self.loss_fn = None
        match self.loss_type: 
            case "mse": 
                self.loss_fn = nn.MSELoss()
            case "ce": 
                self.loss_fn = nn.CrossEntropyLoss()
            case "bce": 
                self.loss_fn = nn.BCELoss()

        self.model.to(self.device)

    # ---------------------------------------------------------------------
    def _step(self, batch, is_train=True, stage = 1):
        """one optimisation / validation step"""
        u, x = batch            # unpack dataloader tuple

        
        u = u.to(self.device).float()
        x = x.to(self.device).float()

        if is_train: self.optimizer.zero_grad()

        out = self.model(u)
        total = self.loss_fn(out.reshape(-1, 3), x.reshape(-1, 3))

        if is_train:
            total.backward()

        self.optimizer.step()
        return total.item()

    # ---------------------------------------------------------------------
    def train_loop(self,
                   train_loader,
                   val_loader,
                   epochs=20,
                   early_stop=100, log_path=None, 
                   device = None, 
                   ):

        if log_path is None:
            log_path = self.log_path
        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")
        self.best_val = float("inf")
        if device is not None: 
            self.model.to(device)
            # self.optimizer.to(device)
            old_device = self.device
            self.device = device
            
                                              
        for ep in range(epochs):
            # -------- training ----------
            self.model.train()
            for i,batch in enumerate(train_loader):
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)          # move exp_avg etc.
                tot = self._step(batch, is_train=True)
                global_step = ep*len(train_loader)+i
                self.writer.add_scalar("train/total",  tot, global_step)
                print(f"Epoch {ep} - Step {i} - train loss: {tot:.4f}")

            # -------- validation --------
            self.model.eval()
            with torch.no_grad():
                totals = []
                for batch in val_loader:
                    tot = self._step(batch, is_train=False)
                    totals.append(tot)
                val_mean = sum(totals) / len(totals)
                self.writer.add_scalar("val/total", val_mean, ep)

            # lr scheduler / early stop
            if self.scheduler: self.scheduler.step(val_mean)

            self.save_checkpoint(str(self.check_path) + f"/checkpoint{ep}.pth", ep, val_mean)
            if val_mean < self.best_val:
                self.best_val = val_mean
                self.epochs_no_improve = 0
                self.save_checkpoint(str(self.check_path) + f"/best_model.pth", ep, val_mean)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= early_stop:
                    print(f"Early stopping @epoch {ep}")
                    break
            print(f"Epoch {ep} - val loss: {val_mean:.4f} | best: {self.best_val:.4f}")
            self.device = old_device if device is not None else self.device  # reset device if changed
            self.model.to(self.device)  # ensure model is on the correct device
 


class RTLoss(nn.Module):
    """Though named RTLoss, this one computes both choice loss and rt loss
    
    - for the first model, let's just try to do MSE loss for log(RT) with actual RT
    - in later iterations, maybe we can use ditributional loss to do variational inference. 
        - for example, compute some parameters for let's say the ex normal distribution on each trial and then maximise likelihood of the seen data

    - NOTE: Here, this is restrictive to output = 2
    """
    def __init__(self, rt_dist_specs = {}, choice_loss = None):
        super(RTLoss, self).__init__()
        self.rt_dist_specs = rt_dist_specs
        self.rt_dist = rt_dist_specs.get("dist_name", None)
        if self.rt_dist is not None:
            self.rt_dist = self.rt_dist.lower()
        # If this is none we get MSE log(RT) loss
        if self.rt_dist is None:
            self.rt_loss = nn.MSELoss()
            self.rt_loss_full = nn.MSELoss(reduction='none')
            self.choice_loss = nn.BCELoss()
            self.choice_loss_full = nn.BCELoss(reduction="none")
        elif self.rt_dist == "gaussian": 
            # self.rt_sigma = self.rt_dist_specs.get("sigma", 1)
            # NOTE effectively this would control the weight of RT. 
            # never mind - in the Gaussian setup, RT_weight will be setting this time scale as inverse sigma squared
            self.rt_loss = nn.MSELoss()
            self.rt_loss_full = nn.MSELoss(reduction = "none")
            self.choice_loss = lambda x: -torch.log(x).mean()
            self.choice_loss_full = lambda x: -torch.log(x)
        else:
            raise NotImplementedError("Distributional loss not yet implemented")

    def forward(self, y, labels, full = False):
        """
        y is batch_size, seq_len, 2; so is labels
        defaults to MSE log(RT) loss + BCE choice loss
        """
        if self.rt_dist is None:
            y_log_rt = y[:, :, 1]
            labels_log_rt = torch.log(labels[:, :, 1])
            # filter out the nan values
            mask = ~torch.isnan(labels_log_rt)
            y_log_rt = y_log_rt[mask]
            labels_log_rt = labels_log_rt[mask]
            y_choice = y[:, :, 0][mask]
            labels_choice = labels[:, :, 0][mask]
            # compute RT loss
            if full:
                rt_loss = self.rt_loss_full(y_log_rt, labels_log_rt)
                # compute choice loss
                choice_loss = self.choice_loss_full(y_choice, labels_choice)
            else: 
                rt_loss = self.rt_loss(y_log_rt, labels_log_rt)
                choice_loss = self.choice_loss(y_choice, labels_choice)

        elif self.rt_dist == "gaussian": 
            # NOTE: this is hardly implemented and almost never tried. if you'd like to use this gotta debug. 
            mask = ~torch.isnan(labels[:, :, 1])
            y = y[mask]
            labels = labels[mask] 
            mu_rt = y[..., 0:2] 
            alpha = y[..., 2:]
            rt = labels[..., 1]
            choice = labels[..., 0]
            indices = choice.long().unsqueeze(-1)
            mu_rt_selected = mu_rt.gather(-1, indices).squeeze(-1)
            alpha_selected = alpha.gather(-1, indices).squeeze(-1)
            rt_loss = self.rt_loss(rt, mu_rt_selected)
            choice_loss = self.choice_loss(alpha_selected) # just log! 

        else: 
            raise NotImplementedError

        return rt_loss, choice_loss

class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()
        self.reconstruction_loss = 0
        self.kl_divergence = 0

    def forward(self, c_i, reconstruction, kl_divergence):
        self.reconstruction_loss = reconstruction
        self.kl_divergence = kl_divergence
        return -c_i * self.reconstruction_loss + self.kl_divergence

class RTSlowFastLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bce   = nn.BCELoss()
        self.mse   = nn.MSELoss()
        self.rt_w  = config.get("rt_weight", 0.5)

    def forward(self, model_out, targets, stage = 1):
        """
        model_out : dict { 'rt'          (B,T,1)
                         , 'slow_choice' (B,T,1)
                         , 'fast_choice' (B,T,1) }
        targets   : tuple ( y_true , rt_true )
            y_true : (B,T,1)   binary
            rt_true: (B,T,1)   same scale as out['rt']
        """
        y_true , rt_true = targets              # unpack

        choice_loss, rt_loss = 0, 0
        if stage == 1:
            # choice_loss  = self.bce(model_out['fast_choice'], y_true)
            choice_loss  = self.bce(model_out['slow_choice'], y_true)
            nan_mask = ~torch.isnan(rt_true)
            rt_loss = self.mse(model_out['rt'][nan_mask], rt_true[nan_mask])
            # rt_loss = 0
        else:
            choice_loss  = self.bce(model_out['slow_choice'], y_true)

        # choice_loss= 0.5*(loss_fast + loss_slow) if stage else loss_slow
        choice_loss = choice_loss

        # RT loss
        total   = choice_loss + self.rt_w * rt_loss
        return total, choice_loss, rt_loss


class DynAgentLoss(nn.Module):
    """
    This loss is used to fine-tune a DynAgent model, which takes a ISNNet (I previously used to model a random piece of cortex as a controlled plant) as a reservoir for some rather complicated dynamics.  
    """
    def __init__(self, config): 
        super(DynAgentLoss, self).__init__()
        self.config = config
        self.loss_compound_package = config.get("loss_compound_func_package", "utils.loss_helpers")
        self.loss_compound_str = config.get("loss_compound_func", "interpolate_exp")
        module = importlib.import_module(self.loss_compound_package)
        self.loss_compound_fn = getattr(module, self.loss_compound_str)

        self.weight_regularisation_loss = None
        self.weight_regularisation_weight = config.get("weight_regularisation_weight", 0.01)
        self.output_norm_loss = None
        self.output_norm_weight = config.get("output_norm_weight", 0.01)
        self.input_norm_loss = None
        self.compounded_choice_loss = None
        self.input_norm_weight = config.get("input_norm_weight", 0.01)
        self.N = config.get("N", 10)
        self.loss_compound_vector = self.loss_compound_fn(self.N)
    
    def forward(self, y, y_hat, I_norm, params_norm):
        self.weight_regularisation_loss = self.weight_regularisation_weight * (params_norm)
        self.output_norm_loss = self.output_norm_weight * torch.norm(y)
        self.input_norm_loss = self.input_norm_weight * (I_norm)
        # print((nn.functional.binary_cross_entropy(y_hat, y, reduction = "none").reshape(-1, self.N) @ self.loss_compound_vector.to(y.device)).shape)
        self.compounded_choice_loss = sum(nn.functional.binary_cross_entropy(torch.clamp(y_hat, 1e-7, 1-1e-7), y, reduction = "none").reshape(-1, self.N) @ self.loss_compound_vector.to(y.device))/y.shape[1]
        # print(self.compounded_choice_loss)
        # print(f"Weight regularisation: {self.weight_regularisation_loss.shape}")
        # print(f"Output norm: {self.output_norm_loss.shape}")
        # print(f"Input norm: {self.input_norm_loss.shape}")
        # print(f"Compounded choice: {self.compounded_choice_loss.shape}")

        return self.compounded_choice_loss + self.weight_regularisation_loss + self.output_norm_loss + self.input_norm_loss
    
    def write_loss(self, summary_writer, index, stage): 
        summary_writer.add_scalar(f"{stage} WeightRegularisation", self.weight_regularisation_loss, index)
        summary_writer.add_scalar(f"{stage} OutputNorm", self.output_norm_loss, index)
        summary_writer.add_scalar(f"{stage} InputNorm", self.input_norm_loss, index)
        summary_writer.add_scalar(f"{stage} CompoundedChoice", self.compounded_choice_loss, index)
        summary_writer.add_scalar(f"{stage} Loss", self.weight_regularisation_loss + self.output_norm_loss + self.input_norm_loss + self.compounded_choice_loss, index)

class DynAgentTrainer(Trainer):
    def __init__(self, config=None):
        super(DynAgentTrainer, self).__init__(config)
        self.loss_config = config.get("loss_config") | {"N": self.model.N}
        self.loss_fn = DynAgentLoss(self.loss_config)
        self.task_module = importlib.import_module(config.get("task_module", "tasks.akam_tasks"))
        self.task = getattr(self.task_module, config.get("task", "Two_step"))(**config.get("task_config", {"rew_gen": "walks"}))
    
    def change_task(self, task): 
        self.task = task

    def train(
            self, epochs=10, save_interval=10, reset_interval=200, save_path=None,
            check_path=None, log_path=None, early_stop=100):
        """default is to train on task"""
        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        self.model = self.model.to(self.device)

        self.batch_size = self.config.get("training_config", {}).get("batch_size", 128), 
        # self.mini_batch_size = self.config.get("training_config", {}).get("mini_batch_size", 32)
        self.val_size = round(self.config.get("training_config", {}).get("val_ratio", 0.1) * self.batch_size)
        self.n_trials = self.config.get("training_config", {}).get("n_trials", 100)

        if not hasattr(self, "scheduler"):
            self.scheduler = None
        
        train_indexer = 0
        for e in range(epochs):
            print(f"Epoch {e}")
            self.model.train()
            first_gen = True
            n_repeats = 1
            if e > 10: 
                n_repeats = 1
            if e > 20:
                n_repeats = 10
            if e > 30: 
                n_repeats = 2
            
            for i in range(n_repeats):
                self.optimizer.zero_grad()
                y, y_hat, I_norm = self.model.simulate(self.task, self.batch_size, self.n_trials, reset=first_gen)
                first_gen = False
                if y.isnan().any():
                    print(y.shape)
                    print(y)
                    print("NaN values in y, from training")
                    raise ValueError
                if y_hat.isnan().any():
                    print("NaN values in y_hat, from training")
                    raise ValueError
                params_norm = 0
                for param in self.model.parameters():
                    params_norm += torch.norm(param)
                loss = self.loss_fn(y, y_hat, I_norm, params_norm)
                print(f"Loss: {loss.item()}")
                loss.backward()
                self.optimizer.step()
                self.loss_fn.write_loss(self.writer, train_indexer, "training")
                train_indexer += 1


            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(str(check_path) + "/checkpoint" + str(e) + ".pth", e, loss)

            if (e + 1) % reset_interval == 0 and e / epochs < 0.75:
                if self.scheduler is not None:
                    self._scheduler()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_init

            if self.validate(e, early_stop):
                break
        self.save_model(save_path)


    def validate(self, epoch, early_stop):
        self.model.eval()
        with torch.no_grad():
            y, y_hat, I_norm = self.model.simulate(self.task, self.val_size, self.n_trials)
            if y.isnan().any():
                print(y)
                print("NaN values in y, from val")
                raise ValueError
            if y_hat.isnan().any():
                print("NaN values in y_hat, from val")
                raise ValueError
            
            params_norm = 0
            for param in self.model.parameters():
                params_norm += torch.norm(param)
            loss = self.loss_fn(y, y_hat, I_norm, params_norm)
 
            if self.scheduler is not None:
                self.scheduler.step(loss)               

            self.loss_fn.write_loss(self.writer, epoch, "validation")

            if loss < self.best_val_loss_split:
                self.best_val_loss_split = loss
                self.best_model = deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= early_stop:
                    print("Early stopping")
                    return (1)
            return (0)

class TaskDyVATrainer(Trainer):
    def __init__(self, config=None):
        super(TaskDyVATrainer, self).__init__(config)
        self.loss_fn = ELBO()

    def train(
            self, train_loader, val_loader, epochs=10, save_interval=10, reset_interval=200, save_path=None,
            check_path=None, log_path=None):

        if save_path is None:
            save_path = self.save_path
        if check_path is None:
            check_path = self.check_path
        if log_path is None:
            log_path = self.log_path

        self.writer = SummaryWriter(log_path)
        self.epochs_no_improve = 0
        self.best_val_loss_split = float("inf")

        for e in range(epochs):
            self.model.train()
            for i, (u, x) in enumerate(train_loader):
                u = u.to(self.device)
                x = x.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x, u)
                c_i = (e / epochs)
                loss = self.loss_fn(c_i, self.model.reconstruction, self.model.kl)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("training loss", loss.item(), i + e * len(train_loader))

                # Log additional loss components
                self.writer.add_scalar("training reconstruction loss",
                                       self.loss_fn.reconstruction_loss.mean().item(),
                                       i + e * len(train_loader))
                self.writer.add_scalar("training kl divergence",
                                       self.loss_fn.kl_divergence.mean().item(),
                                       i + e * len(train_loader))

            if (e + 1) % save_interval == 0:
                if check_path is not None:
                    self.save_checkpoint(check_path / f"checkpoint{str(e)}.pth", e, loss)

            if (e + 1) % reset_interval == 0 and e / epochs < 0.75:
                if self.scheduler is not None:
                    self._scheduler()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr_init
                self.epochs_no_improve = 0

            if self.scheduler is None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_init

            if self.validate(val_loader, e):
                break
        self.save_model(save_path)

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = data.to(self.device)
                output = self.model(data)
                c_i = (epoch / self.epochs)
                val_loss = self.loss_fn(c_i, self.model.reconstruction, self.model.kl)
                val_loss = val_loss.mean()
                val_total_loss += val_loss

            avg_val_loss = val_total_loss / len(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)

            self.writer.add_scalar("validation loss", avg_val_loss, epoch)

            # Log additional loss components
            self.writer.add_scalar("validation reconstruction loss",
                                      self.loss_fn.reconstruction_loss.mean().item(),
                                      epoch)
            self.writer.add_scalar("validation kl divergence",
                                      self.loss_fn.kl_divergence.mean().item(),
                                      epoch)

            if avg_val_loss < self.best_val_loss_split:
                self.best_val_loss = avg_val_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= 100:
                    print("Early stopping")
                    return (1)
            return (0)

