import lightning as L
import copy
import hashlib
import torch
import multiprocessing as mp
import traceback
from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
from clearml import Task


from gorq_ml.tuning.utils import ConfigCombinations


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


def _train_model(
    *,
    config: dict,
    current_iteration: int,
    prepared_training: PreparedTraining,
    original_config: dict,
    resume: bool,
) -> dict:
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
            reuse_last_task_id=resume,
            continue_last_task=resume,
        )
        task_existed = False
        config = task.connect_configuration(config, name='config')

    if task_existed:
        logger.warning(f'Task with config {config} already existed, skipping training...')
    else:
        prepared_training.trainer.fit(
            prepared_training.model,
            prepared_training.dl,
            ckpt_path='last' if resume else None
        )
        prepared_training.trainer.validate(
            prepared_training.model,
            dataloaders=prepared_training.dl.val_dataloader()
        )
    task.close()
    logger.critical('Run ended')
    return {
        'task_id': task.id,
        'config': original_config,
    }


def _dispatch_on_device(
        *,
        config: dict,
        device: int,
        num_iterations_todo: int,
        num_iterations_total: int,
        current_iteration: int,
        device_queue: mp.Queue,
        results_queue: mp.Queue,
        get_training_setup_function: Callable[[TrainingConfig, int, str], PreparedTraining],
) -> None:
    logger.warning(f'Starting with {config} on {device}')
    try:
        altered_config = copy.deepcopy(config)
        altered_config['trainer']['max_steps'] = num_iterations_todo
        altered_config['total_iterations'] = num_iterations_total
        altered_config['trainer']['devices'] = [device]

        current_config = TrainingConfig(
            successive_halving_iteration=0,
            config=altered_config,
        )
        prepared_training = get_training_setup_function(
            current_config,
            current_iteration,
            hashlib.sha256(str(config).encode()).hexdigest()
        )
        result = _train_model(
            config=altered_config,
            current_iteration=current_iteration,
            prepared_training=prepared_training,
            original_config=config,
            resume=current_iteration != 0
        )
        results_queue.put(result)
    except Exception as e:
        logger.exception(f'Calculation failed with {e}.\n{traceback.format_exc()}')
        results_queue.put(None)
    finally:
        device_queue.put(device)


def _get_next_population(
    *,
    current_population: list[dict],
    devices: list[int],
    current_iteration: int,
    get_training_setup_function: Callable[[TrainingConfig, int, str], PreparedTraining],
    metric_series: str,
    metric_name: str,
    mode: Literal['min', 'max'],
    population_size: int,
    num_iterations_todo: int,
    num_iterations_total: int,
) -> list[dict]:
    current_results = []
    all_results = []
    with mp.Manager() as manager:
        device_queue = manager.Queue()
        for dev in devices:
            device_queue.put(dev)
        results_queue = manager.Queue()
        for conf in current_population:
            logger.info(f'Waiting for an available device')
            device = device_queue.get(block=True)
            logger.critical(f'Received free device: {device}')
            mp.Process(
                target=_dispatch_on_device,
                kwargs={
                    'config': conf,
                    'device': device,
                    'num_iterations_todo': num_iterations_todo,
                    'num_iterations_total': num_iterations_total,
                    'current_iteration': current_iteration,
                    'device_queue': device_queue,
                    'results_queue': results_queue,
                    'get_training_setup_function': get_training_setup_function,
                },
                daemon=False,
            ).start()
        for _ in range(len(current_population)):
            result = results_queue.get()
            if result is not None:
                all_results.append(result)

    for result in all_results:
        task_id, config = result['task_id'], result['config']
        try:
            task: Task = Task.get_task(task_id)
            latest_metrics = task.get_last_scalar_metrics()
            latest_metric = latest_metrics[metric_name][metric_series]['last']
            current_results.append((config, latest_metric))
        except Exception:
            logger.warning(f'Failed to retrieve result from a run, skipping...')

    for _ in range(5):
        logger.critical('='*20)
    logger.critical('Ended entire pop iteration.')

    sorted_results = sorted(
        current_results,
        key=lambda x: x[1],
        reverse=(mode == 'max')
    )
    return [x[0] for x in sorted_results[:population_size]]


def search(
    *,
    possible_configs: ConfigCombinations,
    get_training_setup_function: Callable[[TrainingConfig, int, str], PreparedTraining],
    metric_series: str,
    metric_name: str,
    mode: Literal['min', 'max'],
    devices: list[int],
    torch_mp_start_method: str = 'spawn'
) -> None:
    torch.multiprocessing.set_start_method(torch_mp_start_method)

    if (len(possible_configs.reduced_population_sizes) + 1) != len(possible_configs.num_iterations):
        raise ValueError("Length of population sizes must be one less than the length of number of iterations.")

    current_population = possible_configs.configs
    total_iterations = 0
    for current_iteration_idx, current_num_iterations in tqdm(
        enumerate(possible_configs.num_iterations),
        total=len(possible_configs.num_iterations),
        desc='Iterating through the search space',
    ):
        total_iterations += current_num_iterations
        if current_iteration_idx == len(possible_configs.num_iterations) - 1:
            next_population_size = 0
        else:
            next_population_size = possible_configs.reduced_population_sizes[current_iteration_idx]
        current_population = _get_next_population(
            current_population=current_population,
            devices=devices,
            current_iteration=current_iteration_idx,
            get_training_setup_function=get_training_setup_function,
            metric_series=metric_series,
            metric_name=metric_name,
            mode=mode,
            population_size=next_population_size,
            num_iterations_todo=current_num_iterations,
            num_iterations_total=sum(possible_configs.num_iterations),
        )
