# import logging
# import pathlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
from ignite.metrics import Metric


class BuildModel:
    """
    The base class to build all models.
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


class Model(BuildModel):
    """
    High Level API model for training PyTorch neural networks.
    """

    def __init__(
        self,
        models: Union[nn.Module, List[nn.Module]],
        optimizers: Union[Optimizer, List[Optimizer]],
        loss_fns: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
        train_handlers: Optional[List[Sequence]] = None,
        validation_handlers: Optional[List[Sequence]] = None,
        training_type: Optional[str] = "SL",  # Supervised Learning (SL) or GAN or SEMI-SUPERVISED-LEARNING (SSL)
        custom_train_step_fn: Optional[Callable] = None,
        custom_validate_step_fn: Optional[Callable] = None,
    ):

        self.train_handlers = train_handlers
        self.validation_handlers = validation_handlers
        self.training_type = training_type
        if self.training_type == "SL":
            self.train_step = self.custom_train_step_fn if not None else self.simple_train_step
            self.val_step = self.custom_validate_step_fn if not None else self.simple_vlidate_step
        elif self.training_type == "GAN"
            self.train_step = self.custom_train_step_fn if not None else self.gan_train_step
            # self.val_step = self.custom_validate_step_fn if not None else self.gan_vlidate_step

        super().__init__(models, optimizers, loss_fns, device)

    def simple_train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        if batch is None:
            raise ValueError("must provide batch data for training")

        self.optimizer.zero_grad()
        X, y = _prepare_batch(batch)
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def gan_train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        d_input = _prepare_batch(batch)
        """
        continue training GAN
        """

    def simple_vlidate_step(self):
        pass

    def fit(self, train_loader: Iterable, num_epochs: Union[int, List[int]] = 10):
        if isinstance(train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_laoder must be a torch DataLoader but got {type(train_loader).__name__}")

        if self._is_list:
            if len(num_epochs) != len(self.models):
                raise ValueError("num_epochs must be the same length as models/optimizers/loss_fns")

        if self.training_type == "SL":
            if self._is_list:
                self._fit_multiple_models(train_loader, num_epochs)
            else:
                self._fit_single_model(train_loader, num_epochs)
        elif self.training_type == "GAN":
            self._fit_gan(train_loader, num_epochs)

    def _fit_single_model(self, train_loader: Iterable, num_epochs: int = 10):
        self.model = self.models
        self.optimizer = self.optimizers
        self.loss_fn = self.loss_fns

        self.model.train()
        self.model = self.model.to(self.device)
        train_engine = Engine(train_step)
        if self.train_handlers:
            for handler in self.train_handlers:
                train_engine.attach(handler)
        train_engine.run(train_loader, num_epochs)

    def _fit_multiple_models(self, train_loader: Iterable, num_epochs: List[int] = 10):
        for i in range(len(self.models)):
            self.model = self.models[i]
            self.optimizer = self.optimizers[i]
            self.loss_fn = self.loss_fns[i]
            train_engine = Engine(train_step)
            if self.train_handlers:
                for handler in self.train_handlers:
                    train_engine.attach(handler)
            train_engine.run(train_loader, num_epochs[i])
            

    def _fit_gan(self, train_loader: Iterable, num_epochs: List[int] = 10):
        # batch_size = self.data_loader.batch_size  # type: ignore
        # g_input = self.g_prepare_batch(batch_size, self.latent_shape, engine.state.device, engine.non_blocking)
        # g_output = self.g_inferer(g_input, self.g_network)

        # # Train Discriminator
        # d_total_loss = torch.zeros(
        #     1,
        # )
        # for _ in range(num_epochs[1]):
        #     self.d_optimizer.zero_grad()
        #     dloss = self.d_loss_function(g_output, d_input)
        #     dloss.backward()
        #     self.d_optimizer.step()
        #     d_total_loss += dloss.item()

        # # Train Generator
        # g_output = self.g_inferer(g_input, self.g_network)
        # self.g_optimizer.zero_grad()
        # g_loss = self.g_loss_function(g_output)
        # g_loss.backward()
        # self.g_optimizer.step()
        pass

    def validate(self, validation_loader: Iterable, metrics: Dict[str, Metric]):
        pass

    def predict(self):
        pass


# #########################################
# ## Example 1 (Supervised Learning)
# ## Single (model/optimizer/loss_fn)
# #########################################
# dataloader = ....
# model = ....
# optimizer = ....
# loss_fn = ....
# model = Model(model, optimizer, loss_fn)
# model.fit(dataloader, num_epochs=100)


# #########################################
# ## Example 2 (Supervised Learning)
# ## Multiple (models/optimizers/loss_fns)
# #########################################
# dataloader = ....
# models = [...., ...., ....]
# optimizers = [...., ...., ....]
# loss_fns = [...., ...., ....]
# model = Model(models, optimizers, loss_fns)
# model.fit(dataloader, num_epochs=[100, 100, 100])


# #########################################
# ## Example 3 (GANs)
# ## Two (models, optimizers, loss_fns)
# #########################################
# # Note: the generator must be passed first in the list

# dataloader = ....

# g = ....
# g_optimizer = ....
# g_loss_fn = ....

# d = ....
# d_optimizer = ....
# d_loss_fn = ....

# models = [g, d]
# optimizers = [g_optimizer, d_optimizer]
# loss_fns = [g_loss_fn, d_loss_fn]
# model = Model(models, optimizers, loss_fns, training_type="GAN")
# model.fit(dataloader, num_epochs=[100, 100])