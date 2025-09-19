import lightning as L
import time
import copy
from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
import multiprocessing as mp
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
    trainer: L.Trainer
    config: dict


def train_model(conf: PreparedTraining, ckpt_path: str | None):
    conf.trainer.fit(
        conf.model,
        conf.dl,
        ckpt_path=ckpt_path
    )


def run_call(
    *,
    config: dict,
    current_iteration: int,
    prepared_training: PreparedTraining,
    past_checkpoints: dict,
    queue: mp.Queue,
    original_config: dict,
) -> None:
    logger.warning(f'Starting a worker')

    existing_tasks: list[Task] = Task.get_tasks(
        project_name=config['task']['project_name'],
        task_name=config['task']['task_name']
    )
    if (
        len(existing_tasks) > 0 and
        existing_tasks[-1].status == Task.TaskStatusEnum.completed and
        (
            existing_tasks[-1]
            .get_configuration_object_as_dict('config')
            .get('search_iteration', current_iteration) == current_iteration # type: ignore
        )
    ):
        task = existing_tasks[-1]
        task_existed = True
    else:
        task = Task.init(
            **config['task'],
        )
        task_existed = False
        config = task.connect_configuration(config, name='config')

    if task_existed:
        logger.warning(f'Task with config {config} already existed, skipping training...')
    else:
        try:
            prepared_training.trainer.fit(
                prepared_training.model,
                prepared_training.dl,
                ckpt_path=past_checkpoints.get(str(config))
            )
            prepared_training.trainer.validate(
                prepared_training.model,
                dataloaders=prepared_training.dl.val_dataloader()
            )
        except Exception as e:
            logger.warning(f'Training failed with e: {e}, continuing...')
    task.close()
    queue.put({
        'task_id': task.id,
        'config': original_config,
    })


def search(
        *,
        possible_configs: list[dict],
        reduced_population_sizes: list[int],
        num_iterations: list[int],
        get_training_setup_function: Callable[[TrainingConfig, int], PreparedTraining],
        metric_series: str,
        metric_name: str,
        mode: Literal['min', 'max'],
        devices: list[int],
) -> None:
    mp.set_start_method('spawn', force=True)
    if (len(reduced_population_sizes) + 1) != len(num_iterations):
        raise ValueError(f'Length of population sizes must be one less than the length of number of iterations.')

    current_population = possible_configs
    past_checkpoints = {}
    total_iterations = 0
    device_processes = {}
    for current_idx, current_iterations in tqdm(
        enumerate(num_iterations),
        total=len(num_iterations),
        desc='Iterating through the search space'
    ):
        total_iterations += current_iterations
        current_results = []
        results_queue = mp.Queue()
        for config in tqdm(
            current_population,
            desc='Iterating through the current possible configs',
            leave=False,
        ):
            original_config = copy.deepcopy(config)
            config['trainer']['max_steps'] = total_iterations
            config['total_iterations'] = sum(num_iterations)
            current_process = None
            prepared_training = None
            while current_process is None:
                for dev_idx in devices:
                    if dev_idx not in device_processes:
                        config['trainer']['devices'] = [dev_idx]
                        current_config = TrainingConfig(
                            successive_halving_iteration=0,
                            config=config,
                        )
                        prepared_training = get_training_setup_function(current_config, current_idx)
                        current_process = mp.Process(
                            target=run_call,
                            kwargs={
                                'config': config,
                                'current_iteration': current_idx,
                                'prepared_training': prepared_training,
                                'past_checkpoints': past_checkpoints,
                                'queue': results_queue,
                                'original_config': original_config,
                            },
                        )
                        current_process.start()
                        device_processes[dev_idx] = current_process
                        break
                    elif not device_processes[dev_idx].is_alive():
                        device_processes[dev_idx].join()
                        del device_processes[dev_idx]
                else:
                    logger.warning(f'All devices are busy now, waiting...')
                    time.sleep(1)

        for dev_idx, process in device_processes.items():
            if process is not None:
                process.join()
        finished_trainings = []
        while not results_queue.empty():
            finished_trainings.append(results_queue.get())

        for task_id, config in finished_trainings:
            task: Task = Task.get_task(task_id)
            latest_metrics = task.get_last_scalar_metrics()
            latest_metric = latest_metrics[metric_name][metric_series]['last']
            current_results.append((config, latest_metric))
            past_checkpoints[str(config)] = list(task.models.values())[-1]

        sorted_results = sorted(current_results, key=lambda x: x[1], reverse=mode=='max')
        current_population = [x[0] for x in sorted_results[:reduced_population_sizes[current_idx]]]
