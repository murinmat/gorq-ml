import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
from torch import nn
from tqdm.auto import tqdm
from typing import Literal
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass(kw_only=True)
class LRSchedulerSetup:
    mode: Literal['step', 'epoch']
    name: str
    scheduler: LRScheduler


@dataclass(kw_only=True)
class TrainingConfig:
    model: nn.Module
    loss_fn: nn.Module
    train_dl: DataLoader
    val_dl: DataLoader | None
    device: str
    optimizers: list[Optimizer]
    lr_schedulers: list[LRSchedulerSetup] | None = None


@dataclass(kw_only=True)
class TrainingEpochResult:
    last_step: int
    model: nn.Module
    train_loss: float
    val_metric_values: dict[str, float]


def train_epoch(
    config: TrainingConfig,
    last_step_number: int,
    *,
    update_interval: int = 50,
    log_writer: SummaryWriter | None = None,
    val_metrics: dict[str, nn.Module] | None = None,
) -> TrainingEpochResult:
    if val_metrics is None:
        val_metrics = {}

    optimizers = config.optimizers
    lr_schedulers = [] if config.lr_schedulers is None else config.lr_schedulers
    config.model.to(config.device)
    current_step = last_step_number
    config.model.train()
    metrics = {}
    train_losses = []
    # Train
    for batch in tqdm(config.train_dl, desc=f'Iterating training dataloader', leave=False):
        inputs = batch[0].to(config.device)
        labels = batch[1].to(config.device)

        for o in config.optimizers:
            o.zero_grad()
        outputs = config.model(inputs)
        loss: torch.Tensor = config.loss_fn(outputs, labels)
        loss.backward()
        for o in optimizers:
            o.step()
        for s in lr_schedulers:
            if s.mode == 'step':
                s.scheduler.step()
        current_step += 1
        train_losses.append(loss.item())
        if current_step % update_interval == 0 and log_writer is not None:
            log_writer.add_scalar('loss/train_step', np.mean(train_losses[-50:]), current_step)
            for s in lr_schedulers:
                if s.mode == 'step':
                    for idx, lr in enumerate(s.scheduler.get_last_lr()):
                        log_writer.add_scalar(f'lr_param_group={idx}/{s.name}', lr, current_step)
    for s in lr_schedulers:
        if s.mode == 'epoch':
            s.scheduler.step()
            if log_writer is not None:
                for idx, lr in enumerate(s.scheduler.get_last_lr()):
                    log_writer.add_scalar(f'lr_param_group={idx}/{s.name}', lr, current_step)
    if log_writer is not None:
        loss = np.mean(train_losses) # type: ignore
        log_writer.add_scalar('loss/train_epoch', current_step)
        metrics['train'] = {'loss': loss}

    # Val
    if config.val_dl is not None:
        val_metric_values = {'loss': []} | {k: [] for k in val_metrics.keys()}
        config.model.eval()
        with torch.no_grad():
            for batch in tqdm(config.val_dl, desc='Iterating val dataloader', leave=False):
                inputs = batch[0].to(config.device)
                labels = batch[1].to(config.device)
                outputs = config.model(inputs)
                loss = config.loss_fn(outputs, labels)
                val_metric_values['loss'].append(loss.item())
                for metric_name, metric_fn in val_metrics.items():
                    val_metric_values[metric_name].append(
                        metric_fn(outputs, labels).item()
                    )
        for k, v in val_metric_values.items():
            if log_writer is not None:
                log_writer.add_scalar(f'{k}/val_epoch', np.mean(v), current_step)
            metrics['val'] = val_metric_values
    else:
        metrics['val'] = {}

    return TrainingEpochResult(
        model=config.model,
        last_step=current_step,
        train_loss=metrics['train']['loss'], # type: ignore
        val_metric_values=metrics['val'], # type: ignore
    )
