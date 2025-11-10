import os
import yaml
import lightning as L
import torch
from typing import Type
from torch.utils.data import Dataset
from clearml import Task
from loguru import logger
from lightning.pytorch import callbacks
from lightning.pytorch import seed_everything


from gorq_ml.data import BaseLightningDataModule


def train(
        *,
        config: dict,
        l_module: Type[L.LightningModule],
        train_ds: Dataset,
        val_ds: Dataset, 
        l_data_module: Type[BaseLightningDataModule] = BaseLightningDataModule,
        strategy: str = 'auto',
        compile: bool = True,
        connect_clearml: bool = True,
        deterministic: bool = True,
) -> None:
    seed_everything(42)
    torch.set_float32_matmul_precision('high')

    dl = l_data_module(
        train_ds=train_ds,
        val_ds=val_ds,
        **config['dataloader'],
    )
    logger.info(f'Train samples: {len(train_ds)}') # type: ignore
    logger.info(f'Val samples: {len(val_ds)}') # type: ignore
    model = l_module(
        **config['model'],
        val_dataset=val_ds,
        train_dataset=train_ds,
        num_training_steps=len(dl.train_dataloader()) * config['trainer']['max_epochs'],
    )
    if compile:
        model = torch.compile(model)

    config['checkpoint']['dirpath'] = os.path.join(
        config['checkpoint']['dirpath'],
        config['task']['project_name'],
        config['task']['task_name'],
    )
    if connect_clearml:
        task: Task = Task.init(**config['task'])
        task.connect(config)
    trainer = L.Trainer(
        **config['trainer'],
        callbacks=[callbacks.ModelCheckpoint(**config['checkpoint'])] + [
            getattr(callbacks, k)(**v) for k, v in config.get('callbacks', {}).items() # type: ignore
        ],
        strategy=strategy,
        deterministic=deterministic,
    )
    trainer.fit(model, dl, ckpt_path=config.get('ckpt_path'))
