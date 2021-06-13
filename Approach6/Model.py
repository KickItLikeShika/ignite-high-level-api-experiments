from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
import ignite.distributed as idist


class Model:
    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[Optimizer, Dict[str, Optimizer]],
        loss_fn: Union[Optimizer, Dict[str, nn.Module]],
        device: Optional[torch.device] = torch.device("cpu"),
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def set_distributed_config(
        self,
        backend: Optional[str] = 'nccl',
        **spawn_kwargs: Any
    ):
        self.distributed_config = {
            "backend": backend,
            "spawn_kwargs": spawn_kwargs
        }

    def set_data(
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
        self.dataloader_config = {
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
        pass

    def fit(self, num_epochs: Optional[int] = 10):
        if 'dataloader_config' not in self.__dict__:
            raise ValueError("must set_data before calling fit")
        dataset = self.dataloader_config["dataset"]
        del self.dataloader_config["dataset"]

        train_engine = Engine(self.train_step)

        if 'distributed_config' in self.__dict__:
            backend = self.distributed_config['backend']
            spawn_kwargs = self.distributed_config['spawn_kwargs']
            del self.distributed_config['backend']
            del self.distributed_config['spawn_kwargs']
            def training(local_rank):
                self._prepare_ddp()
                train_loader = idist.auto_dataloader(dataset, **self.dataloader_config)
                train_engine.run(train_loader, num_epochs)
            with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
                parallel.run(training)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, **self.dataloader_config)
            train_engine.run(train_loader, num_epochs)

    def validate(self):
        pass
    
    def predict(
        self,
        data: torch.Tensor
    ):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(data)
            return prediction