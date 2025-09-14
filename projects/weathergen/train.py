import os
import yaml
from typing import Any
from argparse import ArgumentParser
from clearml import Task
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything

from src import data
from src import models


parser = ArgumentParser(description='Train a WeatherGen model')
parser.add_argument('config', help='Path to the config file.')
parser.add_argument(
    '--name',
    required=True,
    type=str,
    help='Name of the experiment to run.'
)
parser.add_argument(
    '--ckpt',
    type=str,
    required=False,
    help='Path to a checkpoint to restart the training from.',
    default=None,
)

def update_dict(d: dict, expr: str) -> None:
    path, value = expr.split("=", 1)
    keys = path.split("/")

    curr = d
    for k in keys[:-1]:
        curr = curr.setdefault(k, {})

    curr[keys[-1]] = value

def main() -> None:
    seed_everything(42)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError(f'Config path {args.config} not found!')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['task']['task_name'] = args.name
    task = Task.init(
        **config['task'],
        continue_last_task=args.ckpt is not None,
        reuse_last_task_id=args.ckpt is not None,
    )
    config = task.connect(config)
    dl = data.WeatherGenDataModule(
        train_files=data.get_train_files(),
        val_files=data.get_val_files(),
        base_augmentations=data.BASE_AUGMENTATIONS,
        train_augmentations=data.TRAIN_AUGMENTATIONS,
        **config['dataloader'],
    )
    model = getattr(models, config['model_name'])(
        **config['model_kwargs'],
        total_training_steps=(
            (len(dl.train_dataloader().dataset) // config['dataloader']['batch_size'] + 1) # type: ignore
            * config['trainer']['max_epochs']
            / config['trainer']['devices']
        )
    )

    config['checkpoint']['dirpath'] = os.path.join(config['checkpoint']['dirpath'], args.name)
    trainer = Trainer(
        **config['trainer'],
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(**config['checkpoint']),
        ],
        deterministic=True,
    )

    trainer.fit(model, dl, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
