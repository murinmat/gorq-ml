import lightning as L
import copy
import hashlib
import asyncio
from loguru import logger
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Literal
from clearml import Task
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _run_call_executor(
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
        succeeded = True
    else:
        try:
            prepared_training.trainer.fit(
                prepared_training.model,
                prepared_training.dl,
                ckpt_path='last' if resume else None
            )
            prepared_training.trainer.validate(
                prepared_training.model,
                dataloaders=prepared_training.dl.val_dataloader()
            )
            succeeded = True
        except Exception as e:
            logger.warning(f'Training failed with e: {e}, continuing...')
            succeeded = False
    task.close()
    logger.critical('Run ended')
    return {
        'task_id': task.id,
        'config': original_config,
        'succeeded': succeeded,
    }


def _get_next_population(
    *,
    current_population: list[dict],
    devices: list[int],
    current_iteration: int,
    executor: ProcessPoolExecutor,
    get_training_setup_function: Callable[[TrainingConfig, int, str], PreparedTraining],
    metric_series: str,
    metric_name: str,
    mode: Literal['min', 'max'],
    population_size: int,
    num_iterations_todo: int,
    num_iterations_total: int,
) -> list[dict]:
    current_results = []
    futures = []
    device_queue = asyncio.Queue()

    for dev in devices:
        device_queue.put_nowait(dev)

    async def submit_config(config):
        dev_idx = await device_queue.get()
        try:
            altered_config = copy.deepcopy(config)
            altered_config['trainer']['max_steps'] = num_iterations_todo
            altered_config['total_iterations'] = num_iterations_total
            altered_config['trainer']['devices'] = [dev_idx]

            current_config = TrainingConfig(
                successive_halving_iteration=0,
                config=altered_config,
            )
            prepared_training = get_training_setup_function(
                current_config,
                current_iteration,
                hashlib.sha256(str(config).encode()).hexdigest()
            )
            fut = executor.submit(
                _run_call_executor,
                config=altered_config,
                current_iteration=current_iteration,
                prepared_training=prepared_training,
                original_config=config,
                resume=current_iteration != 0
            )
            return fut
        finally:
            device_queue.put_nowait(dev_idx)

    async def run_all():
        tasks = [submit_config(cfg) for cfg in current_population]
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    futures = loop.run_until_complete(run_all())
    loop.close()

    for fut in as_completed(futures):
        result = fut.result()
        task_id, config, succeeded = result['task_id'], result['config'], result['succeeded']
        if not succeeded:
            continue
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
) -> None:

    if (len(possible_configs.reduced_population_sizes) + 1) != len(possible_configs.num_iterations):
        raise ValueError("Length of population sizes must be one less than the length of number of iterations.")

    current_population = possible_configs.configs
    total_iterations = 0

    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        for current_iteration_idx, current_num_iterations in tqdm(
            enumerate(possible_configs.num_iterations),
            total=len(possible_configs.num_iterations),
            desc='Iterating through the search space',
        ):
            total_iterations += current_num_iterations
            current_population = _get_next_population(
                current_population=current_population,
                devices=devices,
                current_iteration=current_iteration_idx,
                executor=executor,
                get_training_setup_function=get_training_setup_function,
                metric_series=metric_series,
                metric_name=metric_name,
                mode=mode,
                population_size=possible_configs.reduced_population_sizes[current_iteration_idx],
                num_iterations_todo=current_num_iterations,
                num_iterations_total=sum(possible_configs.num_iterations),
            )
