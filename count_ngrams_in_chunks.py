"""Collecting counts from chunks"""
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import dbm
from typing import Any

import psutil
from datasets import load_dataset

from pmi_masking.load_dataset import get_tokenizer


MEGA = 2 ** 20


def count_ngrams_in_sequence_update(ngram_counter: Counter[tuple[int, ...], int], sequence: list[int],
                                    max_ngram_size: int) -> None:
    """Updates a counter with how many times each n-gram appears in a single sequence.
    :param ngram_counter: A counter to update.
    :param sequence: a list of tokens.
    :param max_ngram_size: count n-grams up to this length.
    """
    for ngram_size in range(1, max_ngram_size + 1):
        for start_i in range(len(sequence) - ngram_size + 1):
            ngram = tuple(sequence[start_i:start_i + ngram_size])
            ngram_counter[ngram] += 1


def count_ngrams_in_batch(batch: list[list[int]], max_ngram_size: int) -> Counter[tuple[int, ...], int]:
    """Counts how many times each n-gram appears in a batch of sequences,
    and how many n-grams appear in the batch, by n-gram size.
    :param batch: a batch of token sequences.
    :param max_ngram_size: count n-grams up to this size.
    :return: a Counter mapping an n-gram to the number of times it appears in the batch.
    """
    ngram_counter = Counter()
    for sequence in batch:
        count_ngrams_in_sequence_update(ngram_counter, sequence, max_ngram_size)
    return ngram_counter


def __get_memory_stats_mb() -> dict:
    mem = psutil.virtual_memory()
    return {
        'total': mem.total // MEGA,
        'used': mem.used // MEGA,
        'available': mem.available // MEGA
    }


def main():
    # TODO: for now, I'm loading the bookcorpus dataset.
    #   later on, different datasets will be supported and I will have
    #   to pass that as a parameter for the function.
    # TODO: implement this with DuckDB, writing into a duckDB, opening and closing the
    #  connection and such.
    dataset_name = 'bookcorpus'
    split = 'train'
    tokenizer_batch_size = 4_000
    # ngram_count_batch_size = 4_000
    ngram_count_batch_size = 200_000
    # n_samples = None
    # n_samples = 15_000
    n_samples = 74_004_228 // 5  # 20% of the dataset!
    n_workers = 4
    max_ngram_size = 5
    # drop_single_count_ngrams = True
    drop_single_count_ngrams = False
    counter_dir = Path('./counters')
    ngram_of_size_file = counter_dir / 'ngrams_of_size.json'
    save_to_json = True
    # save_to_json = False

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H-%M")
    file_handler = logging.FileHandler(f'count_ngrams_in_chunks_{datetime_str}.log', mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    dataset = load_dataset(dataset_name, split=split)
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    # dataset = dataset.set_format('numpy')
    tokenizer = get_tokenizer()

    def tokenize(batch: dict[str, list]):
        return tokenizer(
            batch['text'],
            add_special_tokens=False,
            # return_tensors='np'
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=n_workers,
    )

    def count_n_tokens(sample: dict[str, Any]) -> dict[str, Any]:
        return {'n_tokens': len(sample['input_ids'])}

    dataset = dataset.map(
        count_n_tokens,
        num_proc=n_workers
    )

    n_examples = len(dataset)
    n_tokens = sum(dataset['n_tokens'])
    num_ngrams_of_size = {
        ngram_size: n_tokens - n_examples * (ngram_size - 1)
        for ngram_size in range(1, max_ngram_size + 1)
    }

    with ngram_of_size_file.open('w') as f:
        json.dump(num_ngrams_of_size, f, indent=4)

    def count_ngrams_in_batch_and_save_to_file(batch: dict[str, list], indices: list[int]) -> None:
        start, end = indices[0], indices[-1]
        logger.info(f'starting samples {start} to {end};'
                    f'memory: {__get_memory_stats_mb()}')
        ngram_counter = count_ngrams_in_batch(
            batch['input_ids'],
            max_ngram_size,
        )
        if drop_single_count_ngrams:
            ngram_counter = dict(filter(lambda item: item[1] > 1, ngram_counter.items()))

        base_filename = f'ngram_counter_{start}-{end}'
        ngram_counter = {','.join(map(str, ngram)): count for ngram, count in ngram_counter.items()}
        if save_to_json:
            counter_file = counter_dir / (base_filename + '.json')
            with open(counter_file, 'w') as f:
                json.dump(ngram_counter, f)
        else:
            # TODO: figure out if i want to use that, or if I even want to...
            raise NotImplementedError

        logger.info(f'finished samples {start} to {end};'
                    f'memory: {__get_memory_stats_mb()};'
                    f'counter size: {sys.getsizeof(ngram_counter) // MEGA}')

    dataset.map(
        count_ngrams_in_batch_and_save_to_file,
        with_indices=True,
        batched=True,
        batch_size=ngram_count_batch_size,
        num_proc=n_workers
    )


if __name__ == '__main__':
    main()
