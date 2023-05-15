"""Collecting counts from chunks"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import psutil
from datasets import Dataset as HuggingfaceDataset
from transformers import PreTrainedTokenizerBase
import pyarrow as pa
import pyarrow.parquet as pq

from src import fields
from src.utils import get_token_field_names, get_module_logger, get_total_ngrams_per_size_file, tokenize_dataset, Ngram

MEGA = 2 ** 20
logger = get_module_logger(__name__)


def get_memory_stats_mb() -> dict:
    mem = psutil.virtual_memory()
    return {
        'total': mem.total // MEGA,
        'used': mem.used // MEGA,
        'available': mem.available // MEGA
    }


def count_ngrams_in_batch(batch: list[list[int]], max_ngram_size: int) -> dict[int, Counter[tuple[int, ...], int]]:
    """Counts how many times each ngram appears in the batch.
    :param batch: a list of token sequences.
    :param max_ngram_size: the maximum size of ngrams to consider.
    :returns: a dictionary mapping from ngram sizes to the counter of ngrams of that size.
        Each counter is a mapping from ngrams (tuple of ints) to the number of times it appears in the batch.
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


def count_total_ngrams_of_size_(input_ids: pa.Array, max_ngram_size: int):
    """Helper function for `count_total_ngrams_of_size` for testing."""
    sequence_lengths = pa.compute.list_value_length(input_ids)
    total_ngrams_of_size = {}
    for ngram_size in range(1, max_ngram_size + 1):
        subtract_constant = ngram_size - 1
        total_ngrams_of_size_per_sample = pa.compute.subtract(sequence_lengths, subtract_constant)
        total_ngrams_of_size_per_sample = pa.compute.max_element_wise(total_ngrams_of_size_per_sample, 0)
        total_ngrams_of_size[ngram_size] = pa.compute.sum(total_ngrams_of_size_per_sample).as_py()
    return total_ngrams_of_size


def count_total_ngrams_of_size(dataset: HuggingfaceDataset, max_ngram_size: int) -> dict[int, int]:
    """Counts the total number of ngrams of every size up to `max_ngram_size`.
    :param dataset: the tokenized dataset (assumed tokens are available in the column `input_ids`
    :param max_ngram_size: the maximal ngram size to be counted.
    :returns: a dictionary mapping ngram size to the total number of ngrams of that size.
    """
    # TODO: will this work when the dataset is larger than memory? I'm not sure. need to test that.
    #   if so we might need to chunk it and aggregate over the chunks
    input_ids = dataset.data.column('input_ids')
    return count_total_ngrams_of_size_(input_ids, max_ngram_size)


def prune_ngrams(ngram_counter: dict[tuple[int, ...], int], filter_ngram_count_threshold: int) -> dict[Ngram, int]:
    """Prunes ngrams that occur less than `filter_ngram_count_threshold`."""
    def filter_func(item: tuple) -> bool:
        ngram, count = item
        return count >= filter_ngram_count_threshold

    ngram_counter = dict(filter(filter_func, ngram_counter.items()))
    return ngram_counter


def convert_ngram_counter_to_pa_table(counter: dict[Ngram, int], ngram_size: int) -> pa.Table:
    """Converts a counter containing counts of ngrams of a given size to a pyarrow table to save it in parquet format.
    :param counter: a mapping between an ngram to its count
    :param ngram_size: the size of the ngrams in this counter
    :returns: a pyarrow table with columns 'token_0', 'token_1', ... (depending on `ngram_size`) and 'count'.
    """
    token_columns = [[] for _ in range(ngram_size)]
    token_fields_names = get_token_field_names(ngram_size)
    count_column = []
    for ngram, count in counter.items():
        for i, token in enumerate(ngram):
            token_columns[i].append(token)
        count_column.append(count)
    table = pa.table(
        [*token_columns, count_column],
        names=[*token_fields_names, fields.COUNT]
    )
    return table


def count_ngrams_in_batches_and_save_to_file(dataset: HuggingfaceDataset, n_workers: int,
                                             ngram_count_batch_size: int, max_ngram_size: int,
                                             filter_ngram_count_threshold: int, save_dir: Path) -> None:
    """Splits the dataset into batches and counts ngrams in each batch. saves result for each batch in a file.
    Resulting files are saved in `save_dir` in a separate subdirectory for each ngram size.
    :param dataset: the dataset to process. assumes that it is already tokenized.
    :param n_workers: number of worker processes
    :param ngram_count_batch_size: the size of the batch to split the dataset into
    :param max_ngram_size: maximum ngram size to consider
    :param filter_ngram_count_threshold: only ngrams with counts (per batch) greater or equal than this threshold will
        be saved.
    :param save_dir: directory to save the resulting files
    """
    def count_ngrams_in_batch_and_save_to_file(batch: dict[str, list], indices: list[int]) -> None:
        start, end = indices[0], indices[-1]
        input_ids = batch['input_ids']
        n_tokens_in_batch = sum(len(sequence) for sequence in input_ids)

        logger.info(f'counting ngrams in samples {start}-{end}, '
                    f'n_tokens_in_batch={n_tokens_in_batch}, '
                    f'memory (MB): {get_memory_stats_mb()}')

        ngram_size_to_counter = count_ngrams_in_batch(input_ids, max_ngram_size)

        for ngram_size, ngram_counter in ngram_size_to_counter.items():
            ngrams_before_prune = len(ngram_counter)
            ngram_counter = prune_ngrams(ngram_counter, filter_ngram_count_threshold)
            ngrams_after_prune = len(ngram_counter)
            ngram_counts_table = convert_ngram_counter_to_pa_table(ngram_counter, ngram_size)
            batch_ngram_counts_file = get_batch_ngram_counts_file(ngram_size, start, end)
            pq.write_table(ngram_counts_table, str(batch_ngram_counts_file))
            batch_ngram_counts_file_size_bytes = batch_ngram_counts_file.stat().st_size

            logger.info(f'ngram_size: {ngram_size}, '
                        f'ngrams_before_prune: {ngrams_before_prune}, '
                        f'ngrams_after_prune: {ngrams_after_prune}, '
                        f'batch_ngram_counts_file_size_bytes: {batch_ngram_counts_file_size_bytes}')

        logger.info(f'finished samples {start}-{end}, '
                    f'counter size (MB): {sys.getsizeof(ngram_size_to_counter) // MEGA}, '
                    f'memory (MB): {get_memory_stats_mb()}')

    def get_batch_ngram_counts_file(ngram_size, start, end):
        save_dir_per_size = save_dir / str(ngram_size)
        save_dir_per_size.mkdir(parents=True, exist_ok=True)
        path = save_dir_per_size / f'count_table_{start}-{end}.parquet'
        return path

    dataset.map(
        count_ngrams_in_batch_and_save_to_file,
        with_indices=True,
        batched=True,
        batch_size=ngram_count_batch_size,
        num_proc=n_workers
    )


def count_ngrams_in_batches(tokenized_dataset: HuggingfaceDataset, save_dir: Path,
                            ngram_count_batch_size: int = 200_000, n_workers: int = None,
                            max_ngram_size: int = 5, filter_ngram_count_threshold: int = 1) -> None:
    """Main function for this module. gets a dataset, tokenizes it, counts how many times each ngram appears in the
    dataset and the total number of ngrams of each size are in the dataset, and saves those to files.
    :param tokenized_dataset: dataset to process. assume it contains text in the 'text' column
    :param save_dir: directory to save the resulting files
    :param ngram_count_batch_size: the batch size to use when counting ngrams
    :param n_workers: the number of worker processes to use.
    :param max_ngram_size: maximal size of ngrams to count
    :param filter_ngram_count_threshold: only ngrams with counts (per batch) greater or equal to this threshold will be
        saved.
    """
    if 'input_ids' not in tokenized_dataset.features:
        raise RuntimeError('Dataset should be tokenized. Feature "input_ids" not found.')

    logger.info('start')
    save_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = os.cpu_count()
    logger.info(f'n_workers={n_workers}')

    ngram_of_size_file = get_total_ngrams_per_size_file(save_dir)
    total_ngrams_per_size = count_total_ngrams_of_size(tokenized_dataset, max_ngram_size)
    n_tokens = total_ngrams_per_size[1]
    logger.info(f'n_tokens: {n_tokens}')
    with ngram_of_size_file.open('w') as f:
        json.dump(total_ngrams_per_size, f, indent=4)

    count_ngrams_in_batches_and_save_to_file(tokenized_dataset, n_workers, ngram_count_batch_size,
                                             max_ngram_size, filter_ngram_count_threshold, save_dir)

    logger.info('end')
