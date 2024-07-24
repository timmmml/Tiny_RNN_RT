"""This module wraps the specifications of the training process for the agents


"""
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from copy import deepcopy


def reinitialise_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            reinitialise_weights(layer)


class Trainer:
    def __init__(self, config=None):
        if config is None:
            config = self._default_config()
        self.config = config
        self.save_path = config["save_path"]
        self.check_path = config.get("check_path", None)
        self.log_path = config["log_path"]

        self.model_specs = config["model_specs"]  # Should be a dictionary
        self.model = self._model_type(self.model_specs)

        self.device = config["device"]
        self.optimizer_specs = config["optimizer_specs"]
        self.optimizer = self._optimizer_type(self.optimizer_specs)

        self.loss_fn = ELBO()

        # Load a scheduler from config. Config provides a scheduler name and parameters
        if config.get("scheduler", False):
            self._scheduler()
        self.model = self.model.to(self.device)

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
                    self.save_checkpoint(check_path + "\\checkpoint" + str(e) + ".pth", e, loss)

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

class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()
        self.reconstruction_loss = 0
        self.kl_divergence = 0

    def forward(self, c_i, reconstruction, kl_divergence):
        self.reconstruction_loss = reconstruction
        self.kl_divergence = kl_divergence
        return -c_i * self.reconstruction_loss + self.kl_divergence
