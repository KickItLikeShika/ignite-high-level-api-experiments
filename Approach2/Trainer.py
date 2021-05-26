# import logging
# import pathlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine
from ignite.metrics import Metric


class BuildModel:
    
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optimizer: Union[Optimizer, List[Optimizer]],
        loss_fn: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
    ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self._is_gan = False
        if isinstance(self.model, list) or isinstance(self.optimizer, list) or isinstance(self.loss_fn, list):
            self._is_gan = True
        
        self.build_attributes()
        
    def build_attributes(self):
        self._build_device()
        
        self._build_optimizer()
        
        self._build_model()
        
        self._build_loss_fn()
        
        if self._is_gan:
            self._build_gan()

    def _build_device(self):
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, but found {type(self.device).__name__}")

    def _build_model(self):
        if isinstance(self.model, list):
            if len(self.model) != 2:
                raise ValueError("list of models must contain 2 models only")
            if not isinstance(self.model[0], nn.Module) and not isinstance(self.model[1], nn.Module):
                raise TypeError(
                        "the 2 models must be of type toch.nn.Module, "
                        f"but found {type(self.model).__name__}"
                    )
        elif not isinstance(self.model, nn.Module):
            raise TypeError(
                "model must be of type nn.Module, " f"but found {type(self.model).__name__}"
            )

    def _build_optimizer(self):
        if isinstance(self.optimizer, list):
            if len(self.optimizer) != 2:
                raise ValueError("list of optimizers must contain 2 optimizers only")
            if not isinstance(self.optimizer[0], Optimizer): and not isinstance(self.optimizer[1], Optimizer):
                raise TypeError(
                        "the 2 optimizers must be of type torch.optim.Optimizer, "
                        f"but found {type(self.optimizer).__name__}"
                    )
        elif not isinstance(self.optimizer, Optimizer):
            raise TypeError(
                "optimizer must be of type torch.optim.Optimizer, " f"but found {type(self.optimizer).__name__}"
            )

    def _build_loss_fn(self):
        if isinstance(self.loss_fn, list):
            if len(self.loss_fn) != 2:
                raise ValueError("list of loss_fns must contain 2 loss_fns only")
            if not isinstance(self.loss_fn[0], nn.Module) and not isinstance(self.loss_fn[1], nn.Module):
                raise TypeError(
                        "the 2 loss_fns must be of type toch.nn.Module, "
                        f"but found {type(self.loss_fn).__name__}"
                    )
        elif not isinstance(self.loss_fn, nn.Module):
            raise TypeError(
                "loss_fn must be of type nn.Module, " f"but found {type(self.loss_fn).__name__}"
            )

    def _build_gan(self):
        if (
            not isinstance(self.model, list)
            or not isinstance(self.optimizer, list)
            or not isinstance(self.loss_fn, list)
        ):
            raise ValueError("model, optimizer, and loss_fns must all be lists")

        if len(self.model) != len(self.optimizer) != len(self.loss_fn):
            raise ValueError("models, optimizer, and loss_fns must be the same length")
        
        if len(self.model) != 2 or len(self.optimizer) != 2 or len(self.loss_fn) != 2:
            raise ValueError("models, optimizer, and loss_fns must be of length 2")


class Model(BuildModel):
    
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optimizer: Union[Optimizer, List[Optimizer]],
        loss_fn: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
        train_handlers: Optional[List[Sequence]] = None,
        validation_handlers: Optional[List[Sequence]] = None,
        custom_train_step_fn: Optional[Callable] = None,
        custom_validate_step_fn: Optional[Callable] = None,
        ):
            self.train_handlers = train_handlers
            self.validation_handlers = validation_handlers

    def simple_train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        pass
    
    def gan_train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        pass
    
    def validate_step(self, engine: Engine, batch: Sequence[torch]):
        pass
    
    def fit(self, train_loader: Iterator, num_epochs: Union[int, List[int]]):
        if isinstance(train_loader, torch.utils.data.DataLoader):
            raise TypeError(f"train_laoder must be a torch DataLoader but got {type(train_loader).__name__}")

        if self._is_gan and not isinstance(num_epochs, list) and not len(num_epochs) == 2:
            raise ValueError("num_epochs must be of length 2")
        
        if self._is_gan:
            self.g = self.model[0]
            self.g_optimizer = self.optimizer[0]
            self.g_loss_fn = self.loss_fn[0]

            self.d = self.model[1]
            self.d_optimizer = self.optimizer[1]
            self.d_loss_fn = self.loss_fn[1]

            train_engine = Engine(self.gan_train_step)
            if self.train_handlers:
                for handler in self.train_handlers:
                    train_engine.attach(handler)
            # train_engine.run(train_loader, num_epochs)
        else:
            train_engine = Engine(self.simple_train_step)
            if self.train_handlers:
                for handler in self.train_handlers:
                    train_engine.attach(handler)
            # train_engein.run(train_loader, num_epochs)
    
    def validate(self, val_loader: Iterator, metrics: Dict[str, Metric]):
        pass
    
    def predict(self, data, torch.Tensor):
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
# ## Example 2 (GANs)
# ## Two (models, optimizers, loss_fns)
# #########################################
# # Note: the generator must be passed first 

# dataloader = ....

# g = ....
# g_optimizer = ....
# g_loss_fn = ....

# d = ....
# d_optimizer = ....
# d_loss_fn = ....

# model = [g, d]
# optimizer = [g_optimizer, d_optimizer]
# loss_fn = [g_loss_fn, d_loss_fn]
# model = Model(model, optimizer, loss_fn)
# model.fit(dataloader, num_epochs=[100, 100])
