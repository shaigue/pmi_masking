import itertools
from collections import Counter, defaultdict
from math import log
from typing import Iterable, Optional

from src.db_implementation import fields
from src.load_dataset import load_and_tokenize_dataset
from src.utils import Ngram, compute_number_of_ngrams_per_size_in_vocab, prune_low_count_ngrams


def count_total_ngrams_per_size(tokenized_samples: list[list[int]], max_ngram_size: int) -> dict[int, int]:
    """Counts how many ngrams of each size there are in the input samples.

    :param tokenized_samples: sequences of tokens
    :param max_ngram_size: the maximal ngram size to consider
    :return: a dictionary mapping from ngram_size to the number of ngrams of that size in the input sequences.
    """
    total_number_of_ngrams_per_size = Counter()
    for tokenized_sample in tokenized_samples:
        for ngram_size in range(1, max_ngram_size + 1):
            if len(tokenized_sample) >= ngram_size:
                total_number_of_ngrams_per_size[ngram_size] += len(tokenized_sample) - ngram_size + 1

    return dict(total_number_of_ngrams_per_size)


def count_ngrams(tokenized_samples: list[list[int]], max_ngram_size: int) -> dict[Ngram, int]:
    """Counts individual ngrams in the input.

    :param tokenized_samples: the tokenized input sequences.
    :param max_ngram_size: the maximal ngram size to consider
    :return: dictionary mapping from ngram to the number of times it appears in the input.
    """
    ngram_to_count = Counter()
    for tokenized_sample in tokenized_samples:
        for ngram_size in range(1, max_ngram_size + 1):
            for start_i in range(len(tokenized_sample) - ngram_size + 1):
                ngram = tuple(tokenized_sample[start_i:start_i+ngram_size])
                ngram_to_count[ngram] += 1
    return dict(ngram_to_count)


def compute_log_likelihood(ngram_to_count: dict[Ngram, int],
                           total_ngrams_per_size: dict[int, int]) -> dict[Ngram, float]:
    """Computes the log likelihood of ngrams.

    :param ngram_to_count: dictionary mapping from ngrams to the number of times that they appear in the corpus.
    :param total_ngrams_per_size: dictionary mapping ngram sizes to the number of ngrams of that size.
    :return: a dictionary mapping ngrams to their log likelihood.
    """
    ngram_to_log_likelihood = {}
    for ngram, ngram_count in ngram_to_count.items():
        ngram_size = len(ngram)
        ngram_probability = ngram_count / total_ngrams_per_size[ngram_size]
        ngram_to_log_likelihood[ngram] = log(ngram_probability)

    return ngram_to_log_likelihood


def iter_ngram_segmentations(ngram: Ngram) -> Iterable[list[Ngram]]:
    """Iterates through all possible segmentations of a given ngram, not including the identity segmentation.
    a segmentation of an ngram is way to split the ngram into smaller ngrams.
    for example, [(1, 2), (3,)] is one possible segmentations of (1, 2, 3).

    :param ngram: ngram to segment.
    :return: iterable that yields the possible segmentation, each is a list with sub-ngrams.
    """
    assert len(ngram) > 1

    ngram_size = len(ngram)
    for n_cut_points in range(1, ngram_size):
        for cut_points in itertools.combinations(range(1, ngram_size), n_cut_points):
            segments = []
            i = 0
            for cut_point in cut_points:
                segment = ngram[i:cut_point]
                segments.append(segment)
                i = cut_point

            segment = ngram[i:]
            segments.append(segment)
            yield segments


def compute_max_segmentation_log_likelihood_sum(ngram_to_log_likelihood: dict[Ngram, float]) -> dict[Ngram, float]:
    """Computes the maximum log likelihood sum over all segmentation of an ngram, for every ngram of size >= 2

    :param ngram_to_log_likelihood: dictionary mapping from ngrams to their log likelihood
    :return: dictionary mapping from ngram to it's max segmentation log likelihood sum value.
    """
    ngram_to_max_segmentation_log_likelihood_sum = {}
    for ngram in ngram_to_log_likelihood.keys():
        if len(ngram) == 1:
            continue
        ngram_to_max_segmentation_log_likelihood_sum[ngram] = max(
            sum(ngram_to_log_likelihood[segment] for segment in segmentation)
            for segmentation in iter_ngram_segmentations(ngram)
        )
    return ngram_to_max_segmentation_log_likelihood_sum


def split_ngrams_by_size(ngrams: Iterable[Ngram]) -> dict[int, list[Ngram]]:
    """Splits the ngrams by their size.

    :param ngrams: iterable that yields ngrams.
    :return: mapping from ngram size to list of ngrams of that size in the input ngrams.
    """
    ngrams_by_size = defaultdict(list)
    for ngram in ngrams:
        ngrams_by_size[len(ngram)].append(ngram)
    return dict(ngrams_by_size)


def compute_max_segmentation_log_likelihood_sum_dynamic_programming(ngram_to_log_likelihood: dict[Ngram, float]) -> \
        dict[Ngram, float]:
    """Dynamic programming version of `compute_max_segmentation_log_likelihood_sum`."""
    ngrams_by_size = split_ngrams_by_size(ngram_to_log_likelihood.keys())
    max_ngram_size = max(ngrams_by_size.keys())
    ngram_to_max_segmentation_log_likelihood = {}

    def compute_sub_ngram_max_value(sub_ngram: Ngram) -> float:
        if len(sub_ngram) == 1:
            return ngram_to_log_likelihood[sub_ngram]
        else:
            return max(ngram_to_log_likelihood[sub_ngram], ngram_to_max_segmentation_log_likelihood[sub_ngram])

    for ngram_size in range(2, max_ngram_size + 1):
        for ngram in ngrams_by_size[ngram_size]:
            ngram_to_max_segmentation_log_likelihood[ngram] = max(
                compute_sub_ngram_max_value(ngram[:split_i]) + compute_sub_ngram_max_value(ngram[split_i:])
                for split_i in range(1, ngram_size)
            )

    return ngram_to_max_segmentation_log_likelihood


def pmi_score(log_likelihood: float, max_segmentation_log_likelihood_sum: float) -> float:
    """Computes the pmi score of an ngram (after pushing the log inside)

    Note that this is the formula mentioned in the repo https://github.com/AI21Labs/pmi-masking and is different
    from the formula mentioned in the paper https://arxiv.org/abs/2010.01825.
    """
    return 2 * log_likelihood - max_segmentation_log_likelihood_sum


def compute_pmi_score(ngram_to_log_likelihood: dict[Ngram, float],
                      ngram_to_max_segmentation_log_likelihood_sum: dict[Ngram, float]) -> dict[Ngram, float]:
    """Computes the pmi scores of ngrams.

    :param ngram_to_log_likelihood: dictionary mapping ngrams to their log_likelihood scores.
    :param ngram_to_max_segmentation_log_likelihood_sum: dictionary mapping ngrams to their maximal segmentation log
        likelihood sum.
    :return: dictionary mapping ngrams to their pmi scores.
    """
    ngram_to_pmi_score = {}
    for ngram in ngram_to_max_segmentation_log_likelihood_sum.keys():
        ngram_to_pmi_score[ngram] = pmi_score(
            ngram_to_log_likelihood[ngram],
            ngram_to_max_segmentation_log_likelihood_sum[ngram]
        )
    return ngram_to_pmi_score


def compute_pmi_masking_vocab(ngram_to_pmi_score: dict[Ngram, float], vocab_size: int,
                              ngram_size_to_vocab_percent: dict[int, float]) -> list[Ngram]:
    """Computes the pmi masking vocabulary.

    :param ngram_to_pmi_score: dictionary mapping ngrams to their pmi scores
    :param vocab_size: the size of the resulting vocabulary.
    :param ngram_size_to_vocab_percent: dictionary mapping ngrams size to the percentage of ngrams of that size in the
        resulting vocabulary.
        for example, ngram_size_to_vocab_percent={2: 30, 3: 30, 4:40} means that the resulting vocabulary will be 30%
        ngrams of size 2, 30% ngrams of size 3 and 40% ngrams of size 4.
    :return: list containing the ngrams selected to go into the masking vocabulary.
    """
    number_of_ngrams_per_size_in_vocab = compute_number_of_ngrams_per_size_in_vocab(ngram_size_to_vocab_percent,
                                                                                    vocab_size)

    # split ngrams by size
    ngrams_list_by_size = defaultdict(list)
    for ngram in ngram_to_pmi_score.keys():
        ngrams_list_by_size[len(ngram)].append(ngram)

    # sort according to pmi scores in descending order
    for ngrams_list in ngrams_list_by_size.values():
        ngrams_list.sort(key=lambda ngram: ngram_to_pmi_score[ngram], reverse=True)

    masking_vocab = []
    for ngram_size, ngrams_list in ngrams_list_by_size.items():
        masking_vocab += ngrams_list[:number_of_ngrams_per_size_in_vocab[ngram_size]]

    return masking_vocab


def run_pipeline_naive(experiment_name: str, dataset_name: str, tokenizer_name: str,
                       max_ngram_size: int, min_count_threshold: int, vocab_size: int,
                       ngram_size_to_vocab_percent: dict[int, float], ngram_count_batch_size: int,
                       min_count_batch_threshold: int, n_workers: int, tokenizer_batch_size: int,
                       n_samples: Optional[int]) -> dict:
    """Computes a pmi masking vocabulary from a tokenized dataset.

    To understand what each parameter does, refer to the function "db_implementation.run_pipeline.run_pipeline"
    :return: a dictionary containing the intermediate results of the pipeline, for testing the db based implementation.
    """
    dataset, tokenizer = load_and_tokenize_dataset(dataset_name=dataset_name, tokenizer_name=tokenizer_name,
                                                   tokenizer_batch_size=tokenizer_batch_size, n_samples=n_samples)
    tokenized_samples = dataset['input_ids']
    total_ngrams_per_size = count_total_ngrams_per_size(tokenized_samples, max_ngram_size)
    ngram_to_count = count_ngrams(tokenized_samples, max_ngram_size)
    ngram_to_count = prune_low_count_ngrams(ngram_to_count, min_count_threshold)
    ngram_to_log_likelihood = compute_log_likelihood(ngram_to_count, total_ngrams_per_size)
    ngram_to_max_segmentation_log_likelihood_sum = compute_max_segmentation_log_likelihood_sum(ngram_to_log_likelihood)
    ngram_to_pmi_score = compute_pmi_score(ngram_to_log_likelihood, ngram_to_max_segmentation_log_likelihood_sum)
    pmi_masking_vocab = compute_pmi_masking_vocab(
        ngram_to_pmi_score,
        vocab_size,
        ngram_size_to_vocab_percent,
    )
    return {
        'total_ngrams_per_size': total_ngrams_per_size,
        fields.COUNT: ngram_to_count,
        fields.LOG_LIKELIHOOD: ngram_to_log_likelihood,
        fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM: ngram_to_max_segmentation_log_likelihood_sum,
        fields.PMI_SCORE: ngram_to_pmi_score,
        'pmi_masking_vocab': pmi_masking_vocab
    }


# ==================================== Functions for testing ===========================================================


def compute_pmi_scores_from_tokenized_samples(tokenized_samples: list[list[int]],
                                              max_ngram_size: int) -> dict[Ngram, float]:
    total_ngrams_per_size = count_total_ngrams_per_size(tokenized_samples, max_ngram_size)
    ngram_to_count = count_ngrams(tokenized_samples, max_ngram_size)
    ngram_to_log_likelihood = compute_log_likelihood(ngram_to_count, total_ngrams_per_size)
    ngram_to_max_segmentation_log_likelihood_sum = compute_max_segmentation_log_likelihood_sum(ngram_to_log_likelihood)
    ngram_to_pmi_score = compute_pmi_score(ngram_to_log_likelihood, ngram_to_max_segmentation_log_likelihood_sum)
    return ngram_to_pmi_score
