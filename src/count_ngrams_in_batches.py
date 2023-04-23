"""Collecting counts from chunks"""
import json
import os
import sys
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Any

import psutil
from datasets import Dataset as HuggingfaceDataset
from transformers import PreTrainedTokenizerBase
import pyarrow as pa
import pyarrow.parquet as pq

from src.get_module_logger import get_module_logger

MEGA = 2 ** 20


# TODO: update/write docstring


def __count_ngrams_in_batch(batch: list[list[int]], max_ngram_size: int) -> \
        dict[int, Counter[tuple[int, ...], int]]:
    """Counts how many times each n-gram appears in a batch of sequences,
    and how many n-grams appear in the batch, by n-gram size.
    :param batch: a batch of token sequences.
    :param max_ngram_size: count n-grams up to this size.
    :return: a Counter mapping an n-gram to the number of times it appears in the batch.
    """
    ngram_size_to_counter = {}
    for ngram_size in range(1, max_ngram_size + 1):
        counter = Counter()
        for sequence in batch:
            for start_i in range(len(sequence) - ngram_size + 1):
                ngram = tuple(sequence[start_i:start_i + ngram_size])
                counter[ngram] += 1
        ngram_size_to_counter[ngram_size] = counter
    return ngram_size_to_counter


def __get_memory_stats_mb() -> dict:
    mem = psutil.virtual_memory()
    return {
        'total': mem.total // MEGA,
        'used': mem.used // MEGA,
        'available': mem.available // MEGA
    }


def __tokenize_dataset(dataset: HuggingfaceDataset, tokenizer: PreTrainedTokenizerBase,
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


def __count_total_ngrams_per_size_and_save_to_file(dataset: HuggingfaceDataset, max_ngram_size: int,
                                                   n_workers: int, save_dir: Path) -> None:
    ngram_of_size_file = save_dir / 'total_ngrams_per_size.json'

    def count_tokens_in_sample(sample: dict[str, Any]) -> dict[str, Any]:
        return {'n_tokens': len(sample['input_ids'])}

    dataset = dataset.map(
        count_tokens_in_sample,
        num_proc=n_workers
    )
    n_examples = len(dataset)
    # TODO: this is not correct if there is a sample with the number of tokens lower than `max_ngram_size`
    n_tokens = sum(dataset['n_tokens'])
    total_ngrams_per_size = {
        ngram_size: n_tokens - n_examples * (ngram_size - 1)
        for ngram_size in range(1, max_ngram_size + 1)
    }
    with ngram_of_size_file.open('w') as f:
        json.dump(total_ngrams_per_size, f, indent=4)


def __count_ngrams_in_batches_and_save_to_file(dataset: HuggingfaceDataset, logger: Logger,
                                               n_workers: int, ngram_count_batch_size: int,
                                               max_ngram_size: int, filter_ngram_count_threshold: int,
                                               save_dir: Path) -> None:
    def count_ngrams_in_batch_and_save_to_file(batch: dict[str, list], indices: list[int]) -> None:
        start, end = indices[0], indices[-1]
        logger.info(f'started working on samples {start}-{end};'
                    f'memory: {__get_memory_stats_mb()}')
        ngram_size_to_counter = __count_ngrams_in_batch(
            batch['input_ids'],
            max_ngram_size,
        )
        for ngram_size, counter in ngram_size_to_counter.items():
            token_columns = [[] for _ in range(ngram_size)]
            token_columns_names = [f'token_{i}' for i in range(1, ngram_size + 1)]
            count_column = []
            for ngram, count in counter.items():
                if count < filter_ngram_count_threshold:
                    continue
                for i, token in enumerate(ngram):
                    token_columns[i].append(token)
                count_column.append(count)

            table = pa.table(
                [*token_columns, count_column],
                names=[*token_columns_names, 'count']
            )
            save_dir_per_size = save_dir / str(ngram_size)
            save_dir_per_size.mkdir(parents=True, exist_ok=True)
            path = save_dir_per_size / f'count_table_{start}-{end}.parquet'
            pq.write_table(table, str(path))

        logger.info(f'finished samples {start} to {end};'
                    f'memory: {__get_memory_stats_mb()};'
                    f'counter size: {sys.getsizeof(ngram_size_to_counter) // MEGA}')

    dataset.map(
        count_ngrams_in_batch_and_save_to_file,
        with_indices=True,
        batched=True,
        batch_size=ngram_count_batch_size,
        num_proc=n_workers
    )


def count_ngrams_in_batches(dataset: HuggingfaceDataset, tokenizer: PreTrainedTokenizerBase,
                            save_dir: Path, tokenizer_batch_size: int = 4_000,
                            ngram_count_batch_size: int = 200_000, n_samples: int = None,
                            n_workers: int = None, max_ngram_size: int = 5,
                            filter_ngram_count_threshold: int = 0) -> None:
    logger = get_module_logger(__file__)
    # TODO: maybe add to logging the name of the dataset?
    logger.info('Starting to count ngrams in batches')

    save_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = os.cpu_count()

    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    dataset = __tokenize_dataset(dataset, tokenizer, n_workers, tokenizer_batch_size)

    __count_total_ngrams_per_size_and_save_to_file(dataset, max_ngram_size, n_workers, save_dir)

    __count_ngrams_in_batches_and_save_to_file(dataset, logger, n_workers, ngram_count_batch_size,
                                               max_ngram_size, filter_ngram_count_threshold, save_dir)
    logger.info('Finished counting ngrams in batches')
