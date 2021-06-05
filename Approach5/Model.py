from typing import Callable, Dict, List, Optional, Sequence, Union, Iterator, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
import ignite.distributed as idist
from ignite.metrics import Accuracy, Loss


class Model:
    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[Optimizer, Dict[str, Optimizer]],
        loss_fn: Union[Optimizer, Dict[str, nn.Module]],
        device: Optional[torch.device] = torch.device("cpu"),
        train_handlers: Optional[Sequence] = None,
        ddp: bool = False,
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_handlers = train_handlers
        self.ddp = ddp

        # check model/optimizer/loss_fn here

    def _check_model(self):
        if (not isinstance(self.model, nn.Module)) or (not isinstance(self.model, dict)):
            raise TypeError("for single model: the model should be torch.nn.Module, "
                            "and for multiple models: the models should "
                            "be dict['model_name': torch.nn.Module], but got "
                            f"{type(self.model).__name__}")

    def _check_optimizer(self):
        if (not isinstance(self.optimizer, Optimizer)) or (not isinstance(self.optimizer, dict)):
            raise TypeError("for single optimizer: the optimizer should be torch.optim.Optimizer, "
                            "and for multiple optimizers: the optimizers should "
                            "be dict['optimizer_name': torch.optim.Optimizer], but got "
                            f"{type(self.optimizer).__name__}")

    def _check_loss_fn(self):
        if (not isinstance(self.loss_fn, nn.Module)) or (not isinstance(self.loss_fn, dict)):
            raise TypeError("for single loss_fn: the loss_fn should be torch.nn.Module, "
                            "and for multiple moloss_fns: the loss_fns should "
                            "be dict['loss_fn_name': torch.nn.nn.Module], but got "
                            f"{type(self.loss_fn).__name__}")

    def _check_device(self):
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, but found {type(self.device).__name__}")

    def _check_data(self):
        if ((self.ddp) and (not isinstance(self.data, dict)) or 
            (not isinstance(self.data, torch.utils.data.DataLoader)) and (not isinstance(self.data, dict))):
            raise TypeError("in this case the data must be a "
                            "dict['dataset': torch.utils.data.Dataset, 'batch_size': int, 'num_workers': int, etc..]")

        if (isinstance(self.data, dict)):
            if self.data['dataset'] is None or not isinstance(self.data['dataset'], torch.utils.data.Dataset):
                raise ValueError("you must provide a dataset key of type torch.utils.data.Dataset")

    def _prepare_ddp(self):
        if isinstance(self.model, dict):
            for k in self.model.keys():
                self.model[k].to(self.device)
        else:
            self.model = self.model.to(self.device)

        if self.ddp:
            if isinstance(self.model, dict):
                for key in self.model.keys():
                    self.model[key] = idist.auto_model(self.model[key])
                for key in self.optimizer.keys():
                    self.optimizer[key] = idist.auto_model(self.optimizer[key])
                for key in self.loss_fn.keys():
                    self.loss_fn[key] = self.loss_fn[key].to(idist.device())
            else:
                self.model = idist.auto_model(self.model)                  
                self.optimizer = idist.auto_optim(self.optimizer)
                self.loss_fn = self.loss_fn.to(self.device)

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

    def fit(
        self,
        data: Union[Iterator, Dict[str, Any]],
        num_epochs: Union[int, Dict[str, int]]
    ):
        self.data = data
        self._check_data()

        train_engine = Engine(self.train_step)

        if self.ddp:
            def training(local_rank):
                self._prepare_ddp()
                dataset = data['dataset']
                del data['dataset']
                data_loader = idist.auto_dataloader(dataset, **data)
                train_engine.run(data_loader, num_epochs)

            with idist.Parallel(backend=None) as parallel:
                parallel.run(training)
        else:
            train_engine.run(data, num_epochs)

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