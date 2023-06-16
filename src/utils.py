import logging
from itertools import chain
from math import floor
from pathlib import Path
from sys import getsizeof

import psutil

import config

Ngram = tuple[int, ...]


def get_available_cpus_count() -> int:
    return len(psutil.Process().cpu_affinity())


def get_log_file():
    log_file = config.PROJECT_ROOT / 'log.log'
    return log_file


def get_module_logger(name: str) -> logging.Logger:
    """Configures a module logger.

    :param name: should be __name__ variable in the module that calls this function.
    :return: logger to be used in that module.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_file = get_log_file()
    file_handler = logging.FileHandler(str(log_file), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_file_size_bytes(file: Path) -> int:
    return file.stat().st_size


def compute_number_of_ngrams_per_size_in_vocab(ngram_size_to_vocab_percent: dict[int, float],
                                               vocab_size: int) -> dict[int, int]:
    """Computes how many ngrams of each size there should be in the masking vocabulary.

    :param ngram_size_to_vocab_percent: mapping from ngram size to its percent of the final masking vocabulary.
    :param vocab_size: size of the masking vocabulary.
    :return: mapping from ngram size to the number of ngrams of that size in the final masking vocabulary.
    """
    number_of_ngrams_of_size_in_vocab = {}
    for ngram_size, vocab_percent in ngram_size_to_vocab_percent.items():
        number_of_ngrams_of_size_in_vocab[ngram_size] = floor(vocab_size * vocab_percent / 100)

    # take the remaining tokens from the smallest ngram_size ( = 2)
    extra_ngrams = vocab_size - sum(number_of_ngrams_of_size_in_vocab.values())
    number_of_ngrams_of_size_in_vocab[2] += extra_ngrams
    return number_of_ngrams_of_size_in_vocab


def get_memory_stats_str() -> str:
    """Returns a string that contains information about memory usage."""
    mem = psutil.virtual_memory()
    return f'total: {space_str(mem.total)}, used: {space_str(mem.used)}, available: {space_str(mem.available)}'


def prune_low_count_ngrams(ngram_counter: dict[Ngram, int], min_count_threshold: int) -> dict[Ngram, int]:
    """Prunes ngrams that occur less than `min_count_threshold`."""
    ngram_counter = {ngram: count for ngram, count in ngram_counter.items() if count >= min_count_threshold}
    return ngram_counter


def recursive_total_size_bytes(o: object) -> int:
    """Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and their subclasses:
      tuple, list, dict, set and frozenset.

    :param o: object that we want to compute the size of.
    :return: approximate size of the object in bytes.
    """
    def dict_handler(d: dict):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    seen = set()
    default_size = getsizeof(0)

    def sizeof(o):
        if id(o) in seen:
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def time_str(time_seconds: float) -> str:
    """Converts time in seconds to appropriate scale and representing string."""
    if time_seconds < 120:
        return f'{time_seconds:.2f} seconds'
    time_minutes = time_seconds / 60
    if time_minutes < 120:
        return f'{time_minutes:.2f} minutes'
    time_hours = time_minutes / 60
    if time_hours < 50:
        return f'{time_hours:.2f} hours'
    time_days = time_hours / 24
    return f'{time_days:.2f} days'


def space_str(space_bytes: float) -> str:
    """Converts space in bytes to appropriate scale and generates a representing string."""
    if space_bytes < 2 ** 10:
        return f'{space_bytes:.2f} Bytes'
    space_kb = space_bytes / (2 ** 10)
    if space_kb < 2 ** 10:
        return f'{space_kb:.2f} KB'
    space_mb = space_kb / (2 ** 10)
    if space_mb < 2 ** 10:
        return f'{space_mb:.2f} MB'
    space_gb = space_mb / (2 ** 10)
    if space_gb < 2 ** 10:
        return f'{space_gb:.2f} GB'
    space_tb = space_gb / (2 ** 10)
    return f'{space_tb:.2f} TB'
