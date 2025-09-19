import os
import yaml
import lightning as L
import uuid
import hashlib
from copy import deepcopy
import itertools
from pathlib import Path
from loguru import logger
from typing import Tuple, Any
from argparse import ArgumentParser
from clearml import Task
from lightning.pytorch import callbacks
from lightning.pytorch import seed_everything
from sklearn.model_selection import train_test_split

from gorq_ml.tuning.successive_halving import search, PreparedTraining, TrainingConfig

from src import data
from src import models

parser = ArgumentParser(description='Train a WeatherGen model')
parser.add_argument(
    '--base-config',
    help='Path to the base config file.',
    required=True
)
parser.add_argument(
    '--config-options',
    help='Path to the config file containing all the options',
    required=True
)
parser.add_argument(
    '--name',
    required=True,
    type=str,
    help='Base name of the experiments to run.'
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


def get_train_val_data() -> Tuple[list[Path], list[Path]]:
    all_data = data.load_all_images()
    train, val = train_test_split(all_data, train_size=0.9, random_state=42)
    return train, val


def dict_product(d):
    if not isinstance(d, dict):
        return [d]

    keys = list(d.keys())
    values_lists = []

    for k in keys:
        v = d[k]
        if isinstance(v, dict):
            values_lists.append(dict_product(v))
        elif isinstance(v, list):
            values_lists.append(v)
        else:
            values_lists.append([v])

    combinations = []
    for prod in itertools.product(*values_lists):
        combo = {}
        for key, val in zip(keys, prod):
            combo[key] = val
        combinations.append(combo)

    return combinations


def construct_configs(base: dict, optional: dict) -> Tuple[list[dict], list[int], list[int]]:
    all_configs = []

    for config in optional['configs']:
        expanded_options = dict_product(config)
        for option in expanded_options:
            new_config = deepcopy(base)
            def merge(d1, d2):
                for k, v in d2.items():
                    if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                        merge(d1[k], v)
                    else:
                        d1[k] = v
            merge(new_config, option)
            new_config['config_hash'] = hashlib.sha256(str(new_config).encode()).hexdigest()
            all_configs.append(new_config)

    return all_configs, optional['num_epochs'], optional['reduced_population_sizes']


def setup_training(prepared_config: TrainingConfig) -> PreparedTraining:
    print('='*40)
    print(f'Setting up with: {str(prepared_config.config)}')
    config = prepared_config.config
    base_name = config['base_task_name']
    task_name = f'{base_name}_iteration={prepared_config.successive_halving_iteration}_{config["config_hash"]}'
    config['task']['task_name'] = task_name
    train_images, val_images = get_train_val_data()
    dl = data.FacesDataModule(
        train_images=train_images,
        val_images=val_images,
        base_augmentations=data.BASE_AUGMENTATIONS,
        train_augmentations=data.TRAIN_AUGMENTATIONS,
        **config['dataloader'],
    )
    logger.info(f'Train samples: {len(dl.train_dataloader().dataset)}') # type: ignore
    logger.info(f'Val samples: {len(dl.val_dataloader().dataset)}') # type: ignore
    model = getattr(models, config['model_name'])(
        **config['model_kwargs'],
        total_training_steps=(
            (len(dl.train_dataloader().dataset) // config['dataloader']['batch_size'] + 1) # type: ignore
            * config['final_max_epochs']
            / config['trainer']['devices']
        )
    )

    config['checkpoint']['dirpath'] = os.path.join(config['checkpoint']['dirpath'], task_name)

    trainer = L.Trainer(
        **config['trainer'],
        callbacks=[callbacks.ModelCheckpoint(**config['checkpoint'])] + [
            getattr(callbacks, k)(**v) for k, v in config.get('callbacks').items() # type: ignore
        ], # type: ignore
        deterministic=True,
    )

    task = Task.init(
        **config['task'],
    )
    config = task.connect(config)
    return PreparedTraining(
        model=model,
        dl=dl,
        task=task,
        trainer=trainer,
    )

def main() -> None:
    seed_everything(42)
    args = parser.parse_args()
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    base_config['base_task_name'] = args.name
    with open(args.config_options, 'r') as f:
        config_options = yaml.safe_load(f)
    all_configs, num_epochs, reduced_pop_sizes = construct_configs(base_config, config_options)
    search(
        possible_configs=all_configs,
        reduced_population_sizes=reduced_pop_sizes,
        num_epochs=num_epochs,
        get_training_setup_function=setup_training,
        metric_name='loss',
        metric_series='val',
        mode='min'
    )


if __name__ == '__main__':
    main()
