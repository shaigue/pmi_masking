import logging
from itertools import chain
from math import floor
from pathlib import Path
from sys import getsizeof

import psutil
from datasets import Dataset as HuggingfaceDataset
from transformers import PreTrainedTokenizerBase

import config

Ngram = tuple[int, ...]


def get_cpu_count() -> int:
    return len(psutil.Process().cpu_affinity())


def get_log_file():
    log_file = config.PROJECT_ROOT / 'log.log'
    return log_file


def get_module_logger(name: str) -> logging.Logger:
    """
    :param name: should be __name__ special variable in the module that calls this
        function.
    """
    # TODO: maybe add some config file to the directory for project
    #  level configurations.
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


def tokenize_dataset(dataset: HuggingfaceDataset, tokenizer: PreTrainedTokenizerBase,
                     n_workers: int, tokenizer_batch_size: int):
    def tokenize(batch: dict[str, list]):
        return tokenizer(batch['text'], add_special_tokens=False)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=n_workers,
    )
    return dataset


def validate_ngram_size_to_vocab_percent(ngram_size_to_vocab_percent: dict[int, float]):
    if 1 in ngram_size_to_vocab_percent.keys():
        raise ValueError('there should not be ngrams of size 1 in the vocabulary. '
                         f'input suggests that there should be {ngram_size_to_vocab_percent[1]}% ngrams of size 1.')

    total_percent = sum(ngram_size_to_vocab_percent.values())
    if total_percent != 100:
        raise ValueError(f'the total percents should sum up to 100. '
                         f'input: {ngram_size_to_vocab_percent} sums to {total_percent}.')


def compute_number_of_ngrams_per_size_in_vocab(ngram_size_to_vocab_percent: dict[int, float], vocab_size: int):
    # compute how much ngrams per size we take
    number_of_ngrams_of_size_in_vocab = {}
    for ngram_size, vocab_percent in ngram_size_to_vocab_percent.items():
        number_of_ngrams_of_size_in_vocab[ngram_size] = floor(vocab_size * vocab_percent / 100)
    # take the extra tokens from the smallest ngram_size (=2)
    extra_ngrams = vocab_size - sum(number_of_ngrams_of_size_in_vocab.values())
    number_of_ngrams_of_size_in_vocab[2] += extra_ngrams
    return number_of_ngrams_of_size_in_vocab


def get_memory_stats_str() -> str:
    mem = psutil.virtual_memory()
    return f'total: {space_str(mem.total)}, used: {space_str(mem.used)}, available: {space_str(mem.available)}'


def prune_low_count_ngrams(ngram_counter: dict[Ngram, int], min_count_threshold: int) -> dict[Ngram, int]:
    """Prunes ngrams that occur less than `min_count_threshold`."""
    ngram_counter = {ngram: count for ngram, count in ngram_counter.items() if count >= min_count_threshold}
    return ngram_counter


def recursive_total_size_bytes(o):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
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
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
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
