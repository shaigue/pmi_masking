import logging
from datetime import datetime
from pathlib import Path

# TODO: make sure that the log directory stays the same no matter how this module is called
LOG_DIRECTORY = Path.cwd().parents[1] / 'logs'


def get_module_logger(module_file: str) -> logging.Logger:
    logger_name = Path(module_file).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file = LOG_DIRECTORY / f'{logger_name}_{datetime_str}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
