from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.engine import Engine, _prepare_batch
import ignite.distributed as idist


class Model:
    def __init__(
        self,
        model: Union[nn.Module, Dict[str, nn.Module]],
        optimizer: Union[Optimizer, Dict[str, Opimizer]],
        loss_fn: Union[Optimizer, Dict[str, nn.Module]],
        device: Optional[torch.device] = None,
        ddp: bool = False,
        # amp_mode: str = False
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.ddp = ddp
        # self.amp_mode = amp_mode

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
        if ((self.ddp) and (not isinstance(self.data, torh.utils.data.Dataset)) or 
            (not isinstance(self.data, torch.utils.data.DataLoader)) and (not isinstance(self.data, dict))):
            raise TypeError("in this case the data must be a "
                            "dict['data': torch.utils.data.Dataset, 'batch_size': int, 'num_workers': int, etc..]")

    def _create_auto_training(self):
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
            self.loss_fn = self.loss_fn.to(idist.device())

    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        X, y = _prepare_batch(batch)
        X = X.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def fit(
        self,
        data: Union[Iterator, Dict[str, Any]],
        num_epochs: Union[int, Dict[str, int]]
    ):
        self.data = data
        self._check_data()
        if self.ddp:
            def training(local_rank):
                self._create_auto_training()
                data = idist.auto_dataloader(**data)
                train_engine = Engine(self.train_step)
                train_engine.run(data, num_epochs)
            with idist.Parallel(backend='nccl') as parallel:
                parallel.run(training)
        else:
            self.model = self.model.to(self.device)
            self.loss_fn = self.loss_fn.to(self.device)
            train_engine = Engine(self.train_step)
            train_engine.run(data, num_epochs)

    def validate(self):
        pass

    def predict(self):
        pass


# #########################################
# ## Example 1 (Supervised Learning)
# ## Single (model/optimizer/loss_fn)
# #########################################
# dataloader = DataLoader(...)
# model = torch.nn.Module
# optimizer = torch.optim.Optimizer
# loss_fn = torch.nn.Module
# model = Model(model, optimizer, loss_fn)
# model.fit(dataloader, num_epochs=100)


# #########################################
# ## Example 2 (GANs)
# ## Two (models, optimizers, loss_fns)
# #########################################
# dataloader = DataLoader(...)

# model = {'generator': torch.nn.Module, 'discriminator': torch.nn.Module}
# optimizer = {'generator': torch.optim.Optimizer, 'discriminator': torch.optim.Optimizer}
# loss_fn = {'generator': torch.nn.Module, 'discriminator': torch.nn.Module}
# num_epochs = {'generator': int, 'discriminator': int}

# class CustomModel(Model):
#     def train_step(self, engine, batch):
        """
        implement a train_step method
        for training gans
        """
        # self.model["generator"] = ...
        # self.optimizer["discriminator"] = ...

# model = CustomModel(model, optimizer, loss_fn)
# model.fit(dataloader, num_epochs=num_epochs)


# #########################################
# ## Example 3 (DDP)
# ## Single (model/optimizer/loss_fn)
# #########################################
# dataset = {'data': torch.utils.data.Dataset, 'batch_size': int, etc..}
# model = torch.nn.Module
# optimizer = torch.optim.Optimizer
# loss_fn = torch.nn.Module
# model = Model(model, optimizer, loss_fn, ddp=True)
# model.fit(dataset, num_epochs=100)


# #########################################
# ## Example 4 (DDP) (GANs)
# ## Two (models/optimizers/loss_fns)
# #########################################
# dataset = {'data': torch.utils.data.Dataset, 'batch_size': int, etc..}

# model = {'generator': torch.nn.Module, 'discriminator': torch.nn.Module}
# optimizer = {'generator': torch.optim.Optimizer, 'discriminator': torch.optim.Optimizer}
# loss_fn = {'generator': torch.nn.Module, 'discriminator': torch.nn.Module}
# num_epochs = {'generator': int, 'discriminator': int}

# class CustomModel(Model):
#     def train_step(self, engine, batch):
        """
        implement a train_step method
        for training gans
        """
        # self.model["generator"] = ...
        # self.optimizer["discriminator"] = ...

# model = CustomModel(model, optimizer, loss_fn, ddp=True)
# model.fit(dataset, num_epochs=num_epochs)