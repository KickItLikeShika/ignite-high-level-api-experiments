from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
import ignite.distributed as idist
from ignite.metrics import Metric


class Model:
    
    def __init__(
        model: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[Optimizer, Dict[str, Optimizer]],
        loss_fn: Union[Optimizer, Dict[str, nn.Module]],
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.train_engine = Engine(self.train_step)
        self.val_engine = Engine(self.val_step)

    def _prepare_ddp(self):
        for key in self.__dict__.keys():
            attr = self.__dict__[key]
            # torch.nn.Module
            if isinstance(attr, nn.Module):
                # loss
                if attr.__module__.startswith('torch.nn.modules.loss'):
                    self.__dict__[key] = self.__dict__[key].to(self.device)
                # model
                else:
                    self.__dict__[key] = idist.auto_model(self.__dict__[key])
            # torch.optim.Optimizer (optimizer)
            if isinstance(attr, Optimizer):
                self.__dict__[key] = idist.auto_optim(self.__dict__[key])

    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        X, y = _prepare_batch(batch)
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def val_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            X, y = _prepare_batch(batch)
            X = X.to(self.device)
            y = y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            return {
                'prediction': y_pred,
                'target': y,
                'loss': loss.item()
            }

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        num_epochs: int = 1,
        metrics: List[Union[Metric, str]] = None,
        metrics_on_train: bool = False,
        train_handlers: Optional[List[Callable]] = None,
        val_callbacks: Optional[List[Callable]] = None
    ):
        pass
    
    def validate(self):
        pass
    