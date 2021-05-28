from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from apex import amp as apex_amp

from ignite.engine import Engine, _prepare_batch
from ignite.metrics import Metric
import ignite.distributed as idist


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

        self.build_atributes()

    def build_atributes(self):
        self._build_device()

        self._build_model()

        self._build_optimizer()

        self._build_loss_fn()
        
        if isinstance(self.model, list):
            self._build_multiple_models()

    def _build_device(self):
        if not isinstance(self.device, torch.device):
            raise TypeError(f"device must be a torch.device, but found {type(self.device).__name__}")

    def _build_model(self):
        if isinstance(self.model, list):
            for model in self.model:
                if not isinstance(model, nn.Module):
                    raise TypeError(
                        "model must be of type list[torch.nn.Module], "
                        f"but found {type(self.model).__name__}"
                    )
        elif not isinstance(self.model, nn.Module):
            raise TypeError("model must be of type nn.Module, " f"but found {type(self.model).__name__}")

    def _build_optimizer(self):
        if isinstance(self.optimizer, list):
            for optimizer in self.optimizer:
                if not isinstance(optimizer, Optimizer):
                    raise TypeError(
                        "optimizer must be of type list[torch.optim.Optimizer], "
                        f"but found {type(self.optimizer).__name__}"
                    )
        elif not isinstance(self.optimizer, Optimizer):
            raise TypeError(
                "optimizer must be of type torch.optim.Optimizer, "
                f"but found {type(self.optimizer).__name__}"
            )

    def _build_loss_fn(self):
        if isinstance(self.loss_fn, list):
            for loss_fn in self.loss_fn:
                if not isinstance(loss_fn, nn.Module):
                    raise TypeError(
                        "loss_fn must be of type list[torch.nn.Module], "
                        f"but found {type(self.loss_fn).__name__}"
                    )
        elif not isinstance(self.loss_fn, nn.Module):
            raise TypeError("loss_fn must be of type nn.Module, " f"but found {type(self.loss_fn).__name__}")

    def _build_mulitple_models(self):
        if (
            not isinstance(self.model, list)
            or not isinstance(self.optimizer, list)
            or not isinstance(self.loss_fn, list)
        ):
            raise ValueError("please provide a single model/optimizer/loss_fn "
                             "or provide a list of models/ list of optimizers/etc.")

        if len(self.model) != len(self.optimizer) != len(self.loss_fn):
            raise ValueError("model, optimizer, and loss_fns must be the same length")


class Model(BuildModel):
    
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optimizer: Union[Optimizer, List[Optimizer]],
        loss_fn: Union[Union[Callable, nn.Module], Union[Callable, List[nn.Module]]],
        device: Optional[Union[str, torch.device]] = None,
        amp_mode: Optional[str] = None,
        ddp: bool = False,
        # train_handlers: Optional[List[Sequence]] = None,
        # validation_handlers: Optional[List[Sequence]] = None, 
    ):
        super().__init__(models, optimizers, loss_fns, device)

        self.ddp = ddp
        self.amp_mode = amp_mode
        if self.amp_mode == 'amp':
            self.scaler = torch.cuda.amp.GradScaler()

        if not isinstance(self.amp_mode, str):
            raise TypeError("amp_mode must be of type str")
        if (self.amp_mode) and (self.amp_mode != 'apex' or self.amp_mode != 'amp'):
            raise ValueError(f"amp_mode must be 'amp' or 'apex', but found {self.amp_mode}")

        if isinstance(self.ddp, bool):
            raise TypeError("ddp must be of type bool")

    def train_step(self, engine: Engine, batch: Sequence[torch.Tensor]):
        self.model.train()
        self.optimizer.zero_grad()
        X, y = _prepare_batch(batch)
        X = X.to(self.device)
        y = y.to(self.device)
        if self.amp_mode == 'amp':
            with autocast(enabled=True):
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            if self.amp_mode == 'apex':
                with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        return {"prediction": y_pred, "target": y, "loss": loss.item()}

    def inferer_step(self):
        pass

    def fit(self, train_loader: Iterable, num_epochs: Union[int, List[int]]):
        """
        NOTES:
            we should check here if we got a list of models
            to allow users to train GANs problery with just
            overriding the `train_step` method
            # if list_of_models:
            # else:
        """
        if self.ddp:
            def training(local_rank):
                self.model = idist.auto_model(self.model)
                self.optimizer = idist.auto_optim(self.optimizer)
                self.loss_fn = self.loss_fn.to(idist.device())
                train_loader = idist.auto_dataloader(train_loader)
                train_engine = Engine(self.train_step)
                train_engine.run(train_loader, num_epochs)

            with idist.Parallel(backend='nccl') as parallel:
                parallel.run(training)
        else:
            self.model = self.model.to(self.device)
            self.loss_fn = self.loss_fn.to(self.device)
            train_engine = Engine(self.train_step)
            # handle the handlers here before start training
            train_engine.run(train_loader, num_epochs)
    
    def validate(self):
        pass

    def predict(self):
        pass
