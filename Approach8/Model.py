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
        loss_fn: Union[Optimizer, Dict[str, nn.Module]],
        ddp: Optional[bool] = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.ddp = ddp

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

    def set_train_data(
        self, 
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        sampler: Optional[torch.utils.data.Sampler] = None,
        batch_sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: Optional[int] = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: Optional[bool] = False,
        drop_last: Optional[bool] = False
    ):
        self.train_loader_config = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last
        }

    def set_validation_data(
        self, 
        dataset: torch.utils.data.Dataset,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        sampler: Optional[torch.utils.data.Sampler] = None,
        batch_sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: Optional[int] = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: Optional[bool] = False,
        drop_last: Optional[bool] = False
    ):
        self.validation_loader_config = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last
        }

    def attach_train_on_event(self, handler, handler_event, *args, **kwargs):
        self.train_engine.add_event_handler(event_name=handler_event, handler=handler, *args, **kwargs)
        return self

    def attach_val_on_event(self, handler, handler_event, *args, **kwargs):
        self.val_engine.add_event_handler(event_name=handler_event, handler=handler, *args, **kwargs)
        return self

    def attach_train(self, handler, name):
        handler.attach(self.train_engine, name)
        return self

    def attach_validation(self, handler, name):
        handler.attach(self.val_engine, name)
        return self

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
        num_epochs: int,
        validation: Optional[bool] = False,
        validate_every: Optional[int] = 1
    ):
        if 'train_loader_config' not in self.__dict__:
            raise ValueError("must set_train_data before calling fit")

        if validation:
            if 'validation_loader_config' not in self.__dict__:
                raise ValueError("must set_validation_data before for validate")
            dataset = self.validation_loader_config['dataset']
            del self.validation_loader_config['dataset']
            val_loader = torch.utils.data.DataLoader(dataset, **self.validation_loader_config)
            self.validation_loader_config['dataset'] = dataset

            def validation_epoch(engine):
                epoch = engine.state.epoch
                val_state = self.val_engine.run(val_loader, 1)
                print(val_state.metrics)

            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED(every=validate_every), validation_epoch)

        dataset = self.train_loader_config['dataset']
        del self.train_loader_config['dataset']

        if self.ddp:
            train_loader = idist.auto_dataloader(dataset, **self.train_loader_config)
            self._prepare_ddp()
        else:
            train_loader = torch.utils.data.DataLoader(dataset, **self.train_loader_config)
        self.train_loader_config['dataset'] = dataset

        self.train_engine.run(train_loader, num_epochs)

    def validate(self):
        if 'validation_loader_config' not in self.__dict__:
            raise ValueError("must set_data before calling validate")
        
        dataset = self.validation_loader_config["dataset"]
        del self.validation_loader_config["dataset"]
        val_loader = torch.utils.data.DataLoader(dataset, **self.dataloader_config)
        self.validation_loader_config['dataset'] = dataset

        self.val_engine.run(val_loader, 1)

    def predict(
        self,
        data: torch.Tensor
    ):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(data)
            return prediction
