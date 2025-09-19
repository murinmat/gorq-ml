import lightning as L
import time
from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
from multiprocessing import Process
from clearml import Task


@dataclass(kw_only=True)
class TrainingConfig:
    successive_halving_iteration: int
    config: dict


@dataclass(kw_only=True)
class TrainedModel:
    checkpoint_path: Path
    config: TrainingConfig
    metric_value: float


@dataclass(kw_only=True)
class PreparedTraining:
    model: L.LightningModule
    dl: L.LightningDataModule
    task: Task
    trainer: L.Trainer


def train_model(conf: PreparedTraining, ckpt_path: str | None):
    conf.trainer.fit(
        conf.model,
        conf.dl,
        ckpt_path=ckpt_path
    )


def search(
        *,
        possible_configs: list[dict],
        reduced_population_sizes: list[int],
        num_epochs: list[int],
        get_training_setup_function: Callable[[TrainingConfig], PreparedTraining],
        metric_series: str,
        metric_name: str,
        mode: Literal['min', 'max'],
) -> None:
    if (len(reduced_population_sizes) + 1) != len(num_epochs):
        raise ValueError(f'Length of population sizes must be one less than the length of number of epochs.')
    
    current_population = possible_configs
    past_checkpoints = {}
    total_epochs = 0
    for current_idx, current_epochs in tqdm(
        enumerate(num_epochs),
        total=len(num_epochs),
        desc='Iterating through the search space'
    ):
        total_epochs += current_epochs
        current_results = []
        for config in tqdm(
            current_population,
            desc='Iterating through the current possible configs',
            leave=False,
        ):
            current_config = TrainingConfig(
                successive_halving_iteration=0,
                config=config,
            )
            prepared_training = get_training_setup_function(current_config)
            prepared_training.trainer.fit_loop.max_epochs = total_epochs
            logger.info(f'Training: {current_config.config}')

            try:
                prepared_training.trainer.fit(
                    prepared_training.model,
                    prepared_training.dl,
                    ckpt_path=past_checkpoints.get(str(config))
                )
            except Exception as e:
                logger.warning(f'Training failed with e: {e}, continuing...')
                prepared_training.task.close()
                continue
            time.sleep(10)
            latest_metrics = prepared_training.task.get_last_scalar_metrics()
            latest_metric = latest_metrics[metric_name][metric_series]['last']
            current_results.append((config, latest_metric))
            past_checkpoints[str(config)] = list(prepared_training.task.models.values())[-1]
            prepared_training.task.close()
        sorted_results = sorted(current_results, key=lambda x: x[1], reverse=mode=='max')
        current_population = [x[0] for x in sorted_results[:reduced_population_sizes[current_idx]]]
