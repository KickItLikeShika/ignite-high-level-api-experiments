from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
import ignite.distributed as idist
from ignite.metrics import Metric


class Model:
    # DONE
        # Validation
        # Setting validation data
        # Attaching metrics and handlers
        # Checkpointing

    # TODO
        # Setting multiple data like example(CycleGAN) -> using Torch Datasets is good for now

    # !! DISCUSS
        # - The idea of creating one engine instead of two `self.engine`. (I don't know how can we handle that yet).
        # - Adding bool arg `train` to the handlers methods, and if train=True, the handler will 
        # be attached for training, if train=False, then the handler for validation.
        # - The metrics should we give them to fit/validate or we keep them separated in their 
        # methods as we do now.

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

        self.train_engine = Engine(self.train_step)
        self.val_engine = Engine(self.val_step)

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
        drop_last: Optional[bool] = False,
        train: Optional[bool] = True
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
            "drop_last": drop_last,
            "train": train
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

    def attach_validate(self, handler, name):
        handler.attach(self.val_engine, name)
        return self

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

    def fit(self, num_epochs: Optional[int] = 10):
        if 'dataloader_config' not in self.__dict__:
            raise ValueError("must set_data before calling fit")
        if not self.dataloader_config['train']:
            raise ValueError("to set the data for training, must "
                            "`train` in set_data to be true")
        dataset = self.dataloader_config["dataset"]
        del self.dataloader_config["dataset"]
        train_tmp = self.dataloader_config["train"]
        del self.dataloader_config["train"]

        if 'distributed_config' in self.__dict__:
            backend = self.distributed_config['backend']
            spawn_kwargs = self.distributed_config['spawn_kwargs']
            del self.distributed_config['backend']
            del self.distributed_config['spawn_kwargs']
            def training(local_rank):
                self._prepare_ddp()
                train_loader = idist.auto_dataloader(dataset, **self.dataloader_config)
                self.train_engine.run(train_loader, num_epochs)
            with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
                parallel.run(training)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, **self.dataloader_config)
            self.train_engine.run(train_loader, num_epochs)
        
        self.dataloader_config["train"] = train_tmp

    def validate(self):
        if 'dataloader_config' not in self.__dict__:
            raise ValueError("must set_data before calling validate")
        if self.dataloader_config['train']:
            raise ValueError("to set the data for validation, must "
                            "`train` in set_data to be false")
        
        dataset = self.dataloader_config["dataset"]
        del self.dataloader_config["dataset"]
        train_tmp = self.dataloader_config["train"]
        del self.dataloader_config["train"]

        val_loader = torch.utils.data.DataLoader(dataset, **self.dataloader_config)
        self.val_engine.run(val_loader, 1)
        self.dataloader_config["train"] = train_tmp

    def predict(
        self,
        data: torch.Tensor
    ):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(data)
            return prediction