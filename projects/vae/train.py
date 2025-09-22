import os
import yaml
import lightning as L
from loguru import logger
from argparse import ArgumentParser
from lightning.pytorch import callbacks
from lightning.pytorch import seed_everything

from gorq_ml.tuning.successive_halving import search, PreparedTraining, TrainingConfig
from gorq_ml.tuning.utils import construct_configs


from src import data
from src.model import VAELightningModel
from src.data.common import VAEDataModule


parser = ArgumentParser(description='Train a WeatherGen model')
parser.add_argument(
    '--base-config',
    help='Path to the base config file.',
    required=True
)
parser.add_argument(
    '--config-combinations',
    help='Path to the config file containing all the combinations',
    required=True
)
parser.add_argument(
    '--name',
    required=True,
    type=str,
    help='Base name of the experiments to run.'
)
parser.add_argument(
    '--devices',
    type=str,
    required=True,
    help='A list of devices to use',
)


def setup_training(
        prepared_config: TrainingConfig,
        search_iteration: int,
        training_id: str,
) -> PreparedTraining:
    print('='*40)
    print(f'Setting up with: {str(prepared_config.config)}')
    config = prepared_config.config
    base_name = config['base_task_name']
    task_name = f'{base_name}_{training_id}'
    config['task']['task_name'] = task_name
    config['search_iteration'] = search_iteration

    dataset = getattr(data, config['data']['dataset_name'])
    train_ds = dataset.get_train_dataset(
        **config['data'].get('common_kwargs', {}),
        **config['data'].get('train_kwargs', {}),
    )
    val_ds = dataset.get_val_dataset(
        **config['data'].get('common_kwargs', {}),
        **config['data'].get('val_kwargs', {}),
    )

    dl = VAEDataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        **config['dataloader'],
    )
    logger.info(f'Train samples: {len(dl.train_dataloader().dataset)}') # type: ignore
    logger.info(f'Val samples: {len(dl.val_dataloader().dataset)}') # type: ignore
    model = VAELightningModel(
        **config['model'],
        total_training_steps=config['total_iterations'],
        val_dataset=dl.val_dataloader().dataset,
    )

    config['checkpoint']['dirpath'] = os.path.join(config['checkpoint']['dirpath'], task_name)

    trainer = L.Trainer(
        **config['trainer'],
        callbacks=[callbacks.ModelCheckpoint(**config['checkpoint'])] + [
            getattr(callbacks, k)(**v) for k, v in config.get('callbacks').items() # type: ignore
        ], # type: ignore
        deterministic=True,
    )

    return PreparedTraining(
        model=model,
        dl=dl,
        trainer=trainer,
        config=config,
    )

def main() -> None:
    seed_everything(42)
    args = parser.parse_args()
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    base_config['base_task_name'] = args.name
    with open(args.config_combinations, 'r') as f:
        combinations = yaml.safe_load(f)
    config_combinations = construct_configs(base_config, combinations)
    search(
        possible_configs=config_combinations,
        get_training_setup_function=setup_training,
        metric_name='loss',
        metric_series='val',
        mode='min',
        devices=[int(x) for x in args.devices.split(',')],
    )


if __name__ == '__main__':
    main()
