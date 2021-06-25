# from ModelCifar import Model
from Model import Model

from datetime import datetime
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.optim as optim
import cifar_utils
from torch.cuda.amp import GradScaler, autocast

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import PiecewiseLinear
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed, setup_logger


def run(
    seed=543,
    data_path="/tmp/cifar10",
    output_path="/tmp/output-cifar10/",
    model="resnet18",
    batch_size=512,
    momentum=0.9,
    weight_decay=1e-4,
    num_workers=12,
    num_epochs=24,
    learning_rate=0.1,
    num_warmup_epochs=4,
    validate_every=3,
    checkpoint_every=1000,
    backend=None,
    resume_from=None,
    log_every_iters=15,
    nproc_per_node=None,
    stop_iteration=None,
    with_clearml=False,
    with_amp=False,
    **spawn_kwargs,
):
    """Main entry to train an model on CIFAR10 dataset.

    Args:
        seed (int): random state seed to set. Default, 543.
        data_path (str): input dataset path. Default, "/tmp/cifar10".
        output_path (str): output path. Default, "/tmp/output-cifar10".
        model (str): model name (from torchvision) to setup model to train. Default, "resnet18".
        batch_size (int): total batch size. Default, 512.
        momentum (float): optimizer's momentum. Default, 0.9.
        weight_decay (float): weight decay. Default, 1e-4.
        num_workers (int): number of workers in the data loader. Default, 12.
        num_epochs (int): number of epochs to train the model. Default, 24.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 0.4.
        num_warmup_epochs (int): number of warm-up epochs before learning rate decay. Default, 4.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 1000.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        log_every_iters (int): argument to log batch loss every ``log_every_iters`` iterations.
            It can be 0 to disable it. Default, 15.
        stop_iteration (int, optional): iteration to stop the training. Can be used to check resume from checkpoint.
        with_clearml (bool): if True, experiment ClearML logger is setup. Default, False.
        with_amp (bool): if True, enables native automatic mixed precision. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node
    if backend == "xla-tpu" and with_amp:
        raise RuntimeError("The value of with_amp should be False if backend is xla")

    def output_transform(output):
        y_pred = output['prediction']
        y = output['target']
        return y_pred, y 

    def run_validation(engine):
        metrics_output = "\n".join([f"\t{k}: {v}" for k, v in engine.state.metrics.items()])
        print(f"Epoch {engine.state.epoch} - \n metrics:\n {metrics_output}")

    model, checkpoint, metrics = create_model(config, run_validation, output_transform, **spawn_kwargs)

    # ------------------------------------
    # Train
    # ------------------------------------
    train(config, model, metrics, run_validation, checkpoint, output_transform)

    # ------------------------------------
    # Validation
    # ------------------------------------
    validate(config, model, run_validation, output_transform)    


def get_dataflow(config):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the dataset
        idist.barrier()

    train_dataset, test_dataset = cifar_utils.get_train_test_datasets(config["data_path"])

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()

    return train_dataset


def create_model(config, run_validation, output_transform, **spawn_kwargs):
    model = cifar_utils.get_model(config["model"])
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )
    criterion = nn.CrossEntropyLoss()
    
    checkpoint = Checkpoint(
        {"model": model, 'optimizer': optimizer},
        DiskSaver("/tmp", require_empty=False),
        n_saved=2,
        score_name="train_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
    )

    metrics = {
        "Accuracy": Accuracy(output_transform=output_transform),
        "Loss": Loss(criterion, output_transform=output_transform),
    }

    model = Model(model, optimizer, criterion)

    model.set_distributed_config(backend=config['backend'], **spawn_kwargs)

    return model, checkpoint, metrics


def train(config, model, metrics, run_validation, checkpoint, output_transform):
    train_dataset = get_dataflow(config)

    train_handlers = [
        (Events.EPOCH_COMPLETED(every=2) | Events.COMPLETED, run_validation),
        (Events.COMPLETED, checkpoint)
    ]

    train_dataloader_config = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], 
                                "pin_memory": True, "shuffle": True, "drop_last": True}

    model.fit(train_dataset=train_dataset, train_dataloader_config=train_dataloader_config, 
                num_epochs=config["num_epochs"], train_handlers=train_handlers, metrics=metrics, metrics_on_train=True)


def validate(config, model, run_validation, output_transform):
    print("\n Training is done \n Validation started")

    train_dataset = get_dataflow(config)

    metrics = {
        "Accuracy": Accuracy(output_transform=output_transform)
    }

    val_handlers = [
        (Events.EPOCH_COMPLETED, run_validation),
    ]

    val_dataloader_config = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], 
                                "pin_memory": True, "shuffle": True, "drop_last": True}

    model.validate(val_dataset=train_dataset, val_dataloader_config=val_dataloader_config, 
                    metrics=metrics, val_handlers=val_handlers)


if __name__ == "__main__":
    fire.Fire({"run": run})

