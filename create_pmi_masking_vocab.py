from argparse import ArgumentParser
from importlib import import_module

import config
from src.db_implementation.run_pipeline import run_pipeline
from src.utils import get_module_logger

logger = get_module_logger(__name__)


def get_experiment_config_options() -> list[str]:
    experiment_config_dir = config.EXPERIMENT_CONFIG_DIR
    config_name_options = [file.stem for file in experiment_config_dir.iterdir() if file.suffix == '.py']
    return config_name_options


if __name__ == '__main__':
    parser = ArgumentParser()
    experiment_config_options = get_experiment_config_options()
    parser.add_argument('--experiment_config', choices=experiment_config_options)
    args = parser.parse_args()
    experiment_config_name = args.experiment_config
    logger.info(f'start experiment_config: {experiment_config_name}')
    experiment_config = import_module(f'experiment_config.{experiment_config_name}').config
    run_pipeline(experiment_config)
    logger.info(f'end experiment_config: {experiment_config_name}')