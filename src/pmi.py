import itertools
import json
import time
from collections import Counter
from collections.abc import Iterable
from io import StringIO
from math import log
from multiprocessing import Pool
from typing import Union

from datasets import IterableDataset

from pmi_masking.load_dataset import load_and_tokenize_bookcorpus_dataset, get_tokenizer

# TODO: the part that is for counting n-grams is mostly complete.
#  Next we need to work on the part that computes the PMI scores.
# TODO: document and refactor the code.
# TODO: after I finish writing this code, and running it on wiki+bookcorpus,
#   I want to ask Zach for a code and design review, before integrating this with his module.
# TODO: optimize. I think that there is still a lot of place to improve performance.
# TODO: I might want to publish this code in a separate repository so I can show it on my portfolio.
# TODO: make sure that I download the dateset without caching it.
# TODO: hide tokenization from PMI-masking vocabulary construction. work in token space only.
# TODO: organize the small scale performance tests.
# TODO: How do I measure progress when I'm iterating through a streaming dataset?
#   maybe Zach can help.
# TODO: I see that there is an nltk function https://www.nltk.org/api/nltk.util.html#nltk.util.ngrams
#  that iterates all the ngrams in a sequence, might want to use that instead my code.
Ngram = tuple[int, ...]


# TODO: delete some of the functions below
def iter_ngram_segmentations(ngram: Ngram) -> Iterable[list[Ngram]]:
    """Iterates through all possible segmentation of an n-gram,
    except for the identity segmentation.
    :param ngram: an n-gram.
    :return: an iterable that yields segmentations. Each segmentation is a list of n-grams
        that compose the entire n-gram.
    """
    assert len(ngram) > 1

    ngram_size = len(ngram)
    for n_cut_points in range(1, ngram_size):
        for cut_points in itertools.combinations(range(1, ngram_size), n_cut_points):
            start_i = 0
            segments = []

            for cut_point in cut_points:
                segment = ngram[start_i:cut_point]
                segments.append(segment)
                start_i = cut_point

            segment = ngram[start_i:]
            segments.append(segment)
            yield segments


# TODO: computing PMI per ngram can be parallelized
def compute_single_ngram_pmi(ngram_prob_dict: dict[Ngram, float], ngram: Ngram,
                             new_version: bool = True) -> float:
    """TODO ...
    :param ngram_prob_dict: TODO ...
    :param ngram: TODO ...
    :param new_version: if to use the new version of the PMI measure.
    """
    ngram_prob = ngram_prob_dict[ngram]
    ngram_log_prob = log(ngram_prob)

    min_pmi = None
    for segmentation in iter_ngram_segmentations(ngram):
        segmentation_log_prob = 0
        for segment in segmentation:
            segment_prob = ngram_prob_dict[segment]
            segment_log_prob = log(segment_prob)
            segmentation_log_prob += segment_log_prob

        if new_version:
            pmi = 2 * ngram_log_prob - segmentation_log_prob
        else:
            pmi = ngram_log_prob - segmentation_log_prob

        if min_pmi is None or pmi < min_pmi:
            min_pmi = pmi

    return min_pmi


def compute_ngram_prob_dict(ngram_counter: dict[Ngram, int],
                            n_ngrams_of_size_counter: dict[int, int]) -> dict[Ngram, float]:
    """Computes n-gram probabilities."""
    ngram_prob_dict = {}
    for ngram, ngram_count in ngram_counter.items():
        ngram_prob_dict[ngram] = ngram_count / n_ngrams_of_size_counter[len(ngram)]
    return ngram_prob_dict


def compute_ngram_pmi_dict(ngram_counter: dict[Ngram, int],
                           n_ngrams_of_size_counter: dict[int, int]) -> dict[Ngram, float]:
    """Computes the PMI scores of the dataset."""
    ngram_prob_dict = compute_ngram_prob_dict(ngram_counter, n_ngrams_of_size_counter)
    ngram_pmi_dict = {}
    for ngram in ngram_prob_dict.keys():
        if len(ngram) > 1:
            ngram_pmi_dict[ngram] = compute_single_ngram_pmi(ngram_prob_dict, ngram)
    return ngram_pmi_dict


def create_pmi_masking_vocab(tokenized_dataset: IterableDataset, max_ngram_size: int,
                             vocab_size: int, min_occurrences_in_corpus: int,
                             ngram_proportion: dict[int, float]) -> list[Ngram]:
    """This is the main function for this module. It receives an iterable dataset and some hyper-parameters and returns
    a masking vocabulary in tokens."""
    if max_ngram_size < 2:
        raise ValueError(f'max_ngram_size should be >= 2, got {max_ngram_size}')
    proportion_sum = sum(ngram_proportion.values())
    if proportion_sum != 1:
        raise ValueError(f'the total proportions should sum up to 1. '
                         f'input: {ngram_proportion} sums to {proportion_sum}.')
    ngram_counter, n_ngrams_of_size_counter = count_ngrams_from_iterable_tokenized_dataset(
        tokenized_dataset,
        max_ngram_size,
    )
    ngram_pmi_scores = compute_ngram_pmi_dict(ngram_counter, n_ngrams_of_size_counter)
    ngrams_list_by_size = {ngram_size: [] for ngram_size in range(2, max_ngram_size + 1)}

    for ngram, ngram_count in ngram_counter.items():
        if len(ngram) > 1 and ngram_count >= min_occurrences_in_corpus:
            ngrams_list_by_size[len(ngram)].append(ngram)

    # sort the ngrams of each size by their pmi scores.
    for ngrams_list in ngrams_list_by_size.values():
        ngrams_list: list
        ngrams_list.sort(key=lambda ngram: ngram_pmi_scores[ngram], reverse=True)

    # take the quantity of ngrams required.
    masking_vocab = []
    for ngram_size, ngrams_list in ngrams_list_by_size.items():
        n_ngrams_of_size = round(ngram_proportion[ngram_size] * vocab_size)
        masking_vocab += ngrams_list[:n_ngrams_of_size]
    return masking_vocab


def main():
    tokenizer = get_tokenizer()
    dataset = load_and_tokenize_bookcorpus_dataset(tokenizer)

    masking_vocab = create_pmi_masking_vocab(
        dataset,
        max_ngram_size=5,
        vocab_size=800_000,
        min_occurrences_in_corpus=10,
        ngram_proportion={2: 0.5, 3: 0.25, 4: 0.125, 5: 0.125},
    )
    masking_vocab = list(map(tokenizer.decode, masking_vocab))
    print(masking_vocab)

    # save result to a file
    file = 'masking_vocab.txt'
    with open(file, 'w') as f:
        for word in masking_vocab:
            f.write(word)
            f.write('\n')


def count_ngrams_in_dataset(dataset: Iterable[list[list[int]]], max_ngram_size: int,
                            n_workers: int) -> tuple[Counter, Counter]:
    # TODO: what I'm doing here is `map-reduce`. maybe there is a pre-made library that
    #   I can use?
    ngram_counter = Counter()
    ngrams_of_size_counter = Counter()
    start_time = time.time()

    with Pool(n_workers) as pool:
        # when I try to map the entire dataset, I get a lot of memory errors.
        # so I want to process the dataset batch by batch.
        for i, batch in enumerate(dataset, 1):
            # TODO: maybe split into more then `n_workers` parts?
            batch_split = []
            split_size = len(batch) // n_workers
            for start_i in range(0, len(batch), split_size):
                batch_split.append(batch[start_i:start_i+split_size])

            map_object = pool.map(count_ngrams_in_batch_wrapper, zip(batch_split, itertools.repeat(max_ngram_size)))
            for ngram_counter_, ngrams_of_size_counter_ in map_object:
                ngram_counter.update(ngram_counter_)
                ngrams_of_size_counter.update(ngrams_of_size_counter_)

            curr_time = time.time()
            print(f'batch number: {i}; elapsed time: {round(curr_time - start_time, 4)} seconds.')

    return ngram_counter, ngrams_of_size_counter


def experiment():
    max_ngram_size = 5
    # TODO: increase
    n_examples = 200_000
    # TODO: find optimal value
    batch_size = 4_000
    # TODO: find optimal value
    n_workers = 4

    tokenizer = get_tokenizer()
    dataset = load_and_tokenize_bookcorpus_dataset(
        tokenizer,
        batch_size=batch_size,
        n_examples=n_examples
    )

    start_time = time.time()
    count_ngrams_in_dataset(dataset, max_ngram_size, n_workers)
    end_time = time.time()
    print(f'total time: {round(end_time - start_time, 4)} seconds.')


def serialize_ngram_counter(counter: Counter) -> str:
    # TODO: document
    lines = []
    for ngram, count in counter.items():
        line = f'{",".join(map(str, ngram))}:{count}'
        lines.append(line)
    return '\n'.join(lines)


def deserialize_ngram_counter(serialized_counter: str) -> Counter:
    # TODO: document
    lines = serialized_counter.splitlines()
    counter = Counter()
    for line in lines:
        ngram_str, count_str = line.split(':')
        count = int(count_str)
        ngram_tokens = ngram_str.split(',')
        ngram = tuple(map(int, ngram_tokens))
        counter[ngram] = count
    return counter


def save_ngram_counter_to_file(counter: Counter, file: Union[str, StringIO]):
    # TODO: document
    serialized_counter = serialize_ngram_counter(counter)
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(serialized_counter)
    else:
        file.write(serialized_counter)


def load_ngram_counter_from_file(file: Union[str, StringIO]):
    # TODO: document
    if isinstance(file, str):
        with open(file, 'r') as f:
            serialized_counter = f.read()
    else:
        serialized_counter = file.read()
    return deserialize_ngram_counter(serialized_counter)


def save_ngrams_of_size_counter_to_file(counter: Counter, file: Union[str, StringIO]):
    # TODO: document
    serialized_counter = json.dumps(counter)
    if isinstance(file, str):
        with open(file, 'w') as f:
            f.write(serialized_counter)
    else:
        file.write(serialized_counter)


def load_ngrams_of_size_counter_from_file(file: Union[str, StringIO]):
    # TODO: document
    if isinstance(file, str):
        with open(file, 'r') as f:
            serialized_counter = f.read()
    else:
        serialized_counter = file.read()

    counter_with_str_keys = json.loads(serialized_counter)
    counter = Counter()
    for key, value in counter_with_str_keys.items():
        counter[int(key)] = value
    return counter


def count_ngrams_in_bookcorpus():
    max_ngram_size = 5
    batch_size = 4_000
    n_workers = 4

    tokenizer = get_tokenizer()
    dataset = load_and_tokenize_bookcorpus_dataset(
        tokenizer,
        batch_size=batch_size,
    )

    ngrams_counter, ngrams_of_size_counter = count_ngrams_in_dataset(dataset, max_ngram_size, n_workers)
    save_ngram_counter_to_file(ngrams_counter, 'counters/ngram_counter')
    save_ngrams_of_size_counter_to_file(ngrams_of_size_counter, 'counters/ngrams_of_size_counter')


if __name__ == '__main__':
    # # TODO: make this a test
    # counter = Counter({(1, 2, 3): 10, (1, 2): 23})
    # file = 'example.counter'
    # save_counter_to_file(counter, file)
    # counter2 = load_counter_from_file(file)
    # print(counter2)
    # experiment()
    count_ngrams_in_bookcorpus()
