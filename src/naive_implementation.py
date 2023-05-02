import itertools
from collections.abc import Iterable
from math import log

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from src.count_ngrams_in_batches import count_ngrams_in_batch, tokenize_dataset

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


def create_pmi_masking_vocab(tokenized_dataset, max_ngram_size: int,
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


def count_ngrams_naive(dataset: Dataset, tokenizer: PreTrainedTokenizerBase,
                       max_ngram_size: int) -> dict[tuple[int, ...], int]:
    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer,
        n_workers=1,
        tokenizer_batch_size=1_000
    )
    tokenized_text = tokenized_dataset['input_ids']
    ngram_counts_per_size = count_ngrams_in_batch(tokenized_text, max_ngram_size)
    ngram_counter = {}
    for ngram_size, ngram_counter_ in ngram_counts_per_size.items():
        ngram_counter.update(ngram_counter_)
    return ngram_counter
