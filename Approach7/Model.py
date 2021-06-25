from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Any, Tuple, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, Events, _prepare_batch
import ignite.distributed as idist
from ignite.metrics import Metric


class Model:
    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[Optimizer, Dict[str, Optimizer]],
        loss_fn: Union[Optimizer, Dict[str, nn.Module]]
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_engine = Engine(self.train_step)
        self.val_engine = Engine(self.val_step)

    def _check_dataloader_config(self, config):
        keys = ["batch_size", "shuffle",
            "sampler", "batch_sampler", "num_workers", 
            "collate_fn", "pin_memory", "drop_last"]

        if not isinstance(config, dict):
            raise TypeError("dataloader_config must be a dict, for example, "
                            "`dataloader_config={'batch_size': int, 'num_workers': int, etc..}`")

        for arg in config.keys():
            if arg not in keys:
                raise ValueError("got unexpected key for dataloader_config, "
                                 f"got {arg}, expected keys {keys}")

    def set_distributed_config(
        self,
        backend: Optional[str] = 'nccl',
        **spawn_kwargs: Any
    ):
        self.distributed_config = {
            "backend": backend,
            "spawn_kwargs": spawn_kwargs
        }

    def _prepare_ddp(self):
        for key in self.__dict__.keys():
            attr = self.__dict__[key]
            # torch.nn.Module
            if isinstance(attr, nn.Module):
                # loss
                if attr.__module__.startswith('torch.nn.modules.loss'):
                    self.__dict__[key] = self.__dict__[key].to(idist.device())
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
        X = X.to(idist.device())
        y = y.to(idist.device())
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def val_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.eval()
        with torch.no_grad():
            X, y = _prepare_batch(batch)
            X = X.to(idist.device())
            y = y.to(idist.device())
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            return {
                'prediction': y_pred,
                'target': y,
                'loss': loss.item()
            }

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        train_dataloader_config: Dict,
        num_epochs: int = 1,
        val_loader: Optional[Iterable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        metrics_on_train: Optional[bool] = False,
        train_handlers: Optional[List[Tuple[Events, Callable]]] = None,
        val_handlers: Optional[List[Tuple[Events, Callable]]] = None
    ):
        self._check_dataloader_config(train_dataloader_config)

        # handlers on events
        if train_handlers:
            for train_handler in train_handlers:
                self.train_engine.add_event_handler(train_handler[0], train_handler[1])
        
        if metrics and metrics_on_train:
            for name, metric in metrics.items():
                metric.attach(self.train_engine, name)
                metric.attach(self.val_engine, name)

        if val_loader:
            if val_handlers:
                for val_handler in val_handlers:
                    self.val_engine.add_event_handler(val_handler[0], val_handler[1])

            # def validation_epoch(engine):
            #     epoch = engine.state.epoch
            #     self.val_engine.run(val_loader, 1)

        if 'distributed_config' in self.__dict__:
            backend = self.distributed_config['backend']
            spawn_kwargs = self.distributed_config['spawn_kwargs']
            def training(local_rank):
                self._prepare_ddp()
                train_loader = idist.auto_dataloader(train_dataset, **train_dataloader_config)
                self.train_engine.run(train_loader, num_epochs)
            with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
                parallel.run(training)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, **train_dataloader_config)
            self.train_engine.run(train_loader, num_epochs)
        
    def validate(
        self,
        val_dataset: torch.utils.data.Dataset,
        val_dataloader_config: Dict,
        metrics: Optional[Dict[str, Metric]] = None,
        val_handlers: Optional[List[Tuple[Events, Callable]]] = None
    ):
        self._check_dataloader_config(val_dataloader_config)
        val_loader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_config)

        if val_handlers:
            for val_handler in val_handlers:
                self.val_engine.add_event_handler(val_handler[0], val_handler[1])
        if metrics:
            for name, metric in metrics.items():
                metric.attach(self.val_engine, name)
        
        self.val_engine.run(val_loader, 1)
    
    def predict(
        self,
        data: torch.Tensor
    ):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(data)
            return prediction
    