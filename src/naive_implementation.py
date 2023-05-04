import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
from math import log, floor

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from src.utils import Ngram, tokenize_dataset


# TODO: let zach code review this.
# TODO: use this to test the other modules
# TODO (optional): create a notebook that uses this module to showcase that the naive implementation does not scale

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
        ngram_to_pmi_score[ngram] = 2 * ngram_to_log_likelihood[ngram] - \
                                    ngram_to_max_segmentation_log_likelihood_sum[ngram]
    return ngram_to_pmi_score


def compute_pmi_masking_vocab(ngram_to_count: dict[Ngram, int], ngram_to_pmi_score: dict[Ngram, float],
                              vocab_size: int, min_count_threshold: int,
                              ngram_size_to_vocab_percent: dict[int, int]) -> list[Ngram]:
    """Computes the pmi masking vocabulary.
    :param ngram_to_count: dictionary mapping ngrams to their counts
    :param ngram_to_pmi_score: dictionary mapping ngrams to their pmi scores
    :param vocab_size: the size of the resulting vocabulary.
    :param min_count_threshold: ngrams that occur less than this value will be filtered out, and not included in the
        final vocabulary.
    :param ngram_size_to_vocab_percent: dictionary mapping ngrams size to the percentage of ngrams of that size in the
        resulting vocabulary.
        for example, ngram_size_to_vocab_percent={2: 30, 3: 30, 4:40} means that the resulting vocabulary will be 30%
        ngrams of size 2, 30% ngrams of size 3 and 40% ngrams of size 4.
    :return: list containing the ngrams selected to go into the masking vocabulary.
    """
    total_percent = sum(ngram_size_to_vocab_percent.values())
    if total_percent != 100:
        raise ValueError(f'the total percents should sum up to 100. '
                         f'input: {ngram_size_to_vocab_percent} sums to {total_percent}.')
    # compute how much ngrams per size we take
    number_of_ngrams_of_size_in_vocab = {}
    for ngram_size, vocab_percent in ngram_size_to_vocab_percent.items():
        number_of_ngrams_of_size_in_vocab[ngram_size] = floor(vocab_size * vocab_percent / 100)

    # take the extra tokens from the smallest ngram_size (=2)
    extra_ngrams = vocab_size - sum(number_of_ngrams_of_size_in_vocab.values())
    number_of_ngrams_of_size_in_vocab[2] += extra_ngrams

    # split ngrams according to size and filter those who do not exceed the minimal count threshold
    ngrams_list_by_size = defaultdict(list)
    for ngram, ngram_count in ngram_to_count.items():
        if len(ngram) > 1 and ngram_count >= min_count_threshold:
            ngrams_list_by_size[len(ngram)].append(ngram)

    # sort according to pmi scores in descending order
    for ngrams_list in ngrams_list_by_size.values():
        ngrams_list.sort(key=lambda ngram: ngram_to_pmi_score[ngram], reverse=True)

    masking_vocab = []
    for ngram_size, ngrams_list in ngrams_list_by_size.items():
        masking_vocab += ngrams_list[:number_of_ngrams_of_size_in_vocab[ngram_size]]

    return masking_vocab


def compute_pmi_masking_vocab_from_tokenized_samples(tokenized_samples: list[list[int]],
                                                     max_ngram_size: int, vocab_size: int, min_count_threshold: int,
                                                     ngram_size_to_vocab_percent: dict[int, int]) -> list[Ngram]:
    """Computes a pmi masking vocabulary from a tokenized dataset.
    :param tokenized_samples: tokenized input sequences
    :param max_ngram_size: maximal size of ngram to consider
    :param vocab_size: size of the output vocabulary
    :param min_count_threshold: filter ngrams that occur less than this value
    :param ngram_size_to_vocab_percent: dictionary mapping ngram sizes to the percent of the output vocabulary
        that will be of that size.
    :return: list of ngrams that where selected to be in the pmi masking vocabulary.
    """
    total_ngrams_per_size = count_total_ngrams_per_size(tokenized_samples, max_ngram_size)
    ngram_to_count = count_ngrams(tokenized_samples, max_ngram_size)
    ngram_to_log_likelihood = compute_log_likelihood(ngram_to_count, total_ngrams_per_size)
    ngram_to_max_segmentation_log_likelihood_sum = compute_max_segmentation_log_likelihood_sum(ngram_to_log_likelihood)
    ngram_to_pmi_score = compute_pmi_score(ngram_to_log_likelihood, ngram_to_max_segmentation_log_likelihood_sum)
    pmi_masking_vocab = compute_pmi_masking_vocab(
        ngram_to_count,
        ngram_to_pmi_score,
        vocab_size,
        min_count_threshold,
        ngram_size_to_vocab_percent,
    )
    return pmi_masking_vocab


# ==================================== Functions for testing ===========================================================


def count_ngrams_from_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase,
                              max_ngram_size: int) -> dict[Ngram, int]:
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        n_workers=1,
        tokenizer_batch_size=1_000
    )
    tokenized_samples = tokenized_dataset['input_ids']
    return count_ngrams(tokenized_samples, max_ngram_size)


def compute_pmi_scores_from_tokenized_samples(tokenized_samples: list[list[int]],
                                              max_ngram_size: int) -> dict[Ngram, float]:
    total_ngrams_per_size = count_total_ngrams_per_size(tokenized_samples, max_ngram_size)
    ngram_to_count = count_ngrams(tokenized_samples, max_ngram_size)
    ngram_to_log_likelihood = compute_log_likelihood(ngram_to_count, total_ngrams_per_size)
    ngram_to_max_segmentation_log_likelihood_sum = compute_max_segmentation_log_likelihood_sum(ngram_to_log_likelihood)
    ngram_to_pmi_score = compute_pmi_score(ngram_to_log_likelihood, ngram_to_max_segmentation_log_likelihood_sum)
    return ngram_to_pmi_score
