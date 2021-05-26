# import logging
# import pathlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
from ignite.metrics import Metric


class BuildTrainer:
    """
    The base class to build all trainers.
    """

    def __init__(
        self,
        models: Union[nn.Module, List[nn.Module]],
        optimizers: Union[Optimizer, List[Optimizer]],
        loss_fns: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
    ):

        self.models = models
        self.optimizers = optimizers
        self.loss_fns = loss_fns
        self.device = device

        self._is_list = False
        if isinstance(self.models, list) or isinstance(self.optimizers, list) or isinstance(self.loss_fns, list):
            self._is_list = True

        self.build_trainer()

    def build_trainer(self):
        # check device
        self._build_device()

        # check model(s)
        self._build_models()

        # check optimizer(s)
        self._build_optimizers()

        # check lists
        if self.is_list:
            self._build_mulitple_models()

    def _build_device(self):
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, but found {type(self.device).__name__}")

    def _build_models(self):
        if isinstance(self.models, list):
            for model in self.models:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "model(s) must be of type toch.nn.Module or list[torch.nn.Module], "
                        f"but found {type(self.models).__name__}"
                    )

        elif not isinstance(self.models, nn.Module):
            raise TypeError(
                "model(s) must be of type nn.Module or list[nn.Module], " f"but found {type(self.models).__name__}"
            )

    def _build_optimizers(self):
        if isinstance(self.optimizers, list):
            for optimizer in self.optimizers:
                if not isinstance(optimizer, Optimizer):
                    raise TypeError(
                        "optimizer(s) must be of type torch.optim.Optimizer "
                        "or list[torch.optim.Optimizer], "
                        f"but found {type(self.optimizers).__name__}"
                    )

        elif not isinstance(self.optimizers, Optimizer):
            raise TypeError(
                "optimizer(s) must be of type torch.optim.Optimizer "
                "or list[torch.optim.Optimizer], "
                f"but found {type(self.optimizers).__name__}"
            )

    def _build_loss_fns(self):
        if isinstance(self.loss_fns, list):
            for model in self.loss_fns:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "loss_fn(s) must be of type toch.nn.Module or list[torch.nn.Module], "
                        f"but found {type(self.loss_fns).__name__}"
                    )

        elif not isinstance(self.loss_fns, nn.Module):
            raise TypeError(
                "loss_fn(s) must be of type nn.Module or list[nn.Module], " f"but found {type(self.loss_fns).__name__}"
            )

    def _build_mulitple_models(self):
        if (
            not isinstance(self.models, list)
            or not isinstance(self.optimizers, list)
            or not isinstance(self.loss_fns, list)
        ):
            raise ValueError("models, optimizer, and loss_fns must all be lists")

        if len(self.models) != len(self.optimizers) != len(self.loss_fns):
            raise ValueError("models, optimizer, and loss_fns must be the same length")


class Trainer(BuildTrainer):
    """
    High Level API Trainer for training PyTorch neural networks.
    """

    def __init__(
        self,
        models: Union[nn.Module, List[nn.Module]],
        optimizers: Union[Optimizer, List[Optimizer]],
        loss_fns: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
        train_handlers: Optional[List[Sequence]] = None,
        validation_handlers: Optional[List[Sequence]] = None,
        learning_type: Optional[str] = "SUPERVISED-LEARNING",  # or GAN or SEMI-SUPERVISED-LEARNING
        custom_train_step_fn: Optional[Callable] = None,
        custom_validate_step_fn: Optional[Callable] = None,
    ):

        self.train_handlers = train_handlers
        self.validation_handlers = validation_handlers
        self.custom_train_step_fn = custom_train_step_fn
        self.custom_validate_step_fn = custom_validate_step_fn

        super().__init__(models, optimizers, loss_fns, device)

    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        if batch is None:
            raise ValueError("must provide batch data for training")

        self.optimizer.zero_grad()
        X, y = _prepare_batch(batch)
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def vlidate_step(self):
        pass

    def fit(self, train_loader: Iterable, num_epochs: Union[int, List[int]] = 10):
        if isinstance(train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_laoder must be a torch DataLoader but got {type(train_loader).__name__}")

        if self._is_list:
            if len(num_epochs) != len(self.models):
                raise ValueError("num_epochs must be the same length as models/optimizers/loss_fns")

        if self._is_list:
            pass
        else:
            self.model = self.models
            self.model.train()
            self.model = self.model.to(self.device)
            self.optimizer = self.optimizers
            self.loss_fn = self.loss_fns
            if self.custom_train_step_fn:
                train_engine = Engine(self.custom_train_step_fn)
            else:
                train_engine = Engine(self.train_step)
            if self.train_handlers:
                for handler in self.train_handlers:
                    train_engine.attach(handler)

            train_engine.run(train_loader, num_epochs)

    def validate(self, validation_loader: Iterable, metrics: Dict[str, Metric]):
        if isinstance(validation_loader, torch.utils.data.DataLoader):
            raise TypeError(f"validation_loader must be a torch DataLoader but got {type(validation_loader).__name__}")

        metrics = metrics or {}
        results = {}

        if self._is_list:
            pass
        else:
            self.model = self.models
            self.optimizer = self.optimizers
            self.loss_fn = self.loss_fns
            self.model = self.model.to(self.device)
            self.model.eval()
            if self.custom_validate_step_fn:
                val_engine = Engine(self.custom_validate_step_fn)
            else:
                val_engine = Engine(self.vlidate_step)
            val_engine.attach(metrics)

    def predict(self):
        pass
