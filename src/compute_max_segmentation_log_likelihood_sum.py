"""This module computes the miniman segments term for the PMI scores with dynamic programming."""


def compute_max_segmentation_log_likelihood_sum():
    # TODO: first, write the algorithm in pure python and test it against the naive implementation.
    # TODO: after that, implement the dynamic algorithm in SQL
    # go over the ngrams with increasing size
    # for each ngram size, we assume that the ngrams of smaller sizes have this value already computed
    # we take the computation from the jupyter notebook and add them.
    # First, I need to initialize the base levels.
    # Secondly, I need to initialize the
    # then, I need to take the maximum of the term
    # when I take the maximal value of some ngram, I should also consider their log_likelihood, since.
    # for level 1, the value is not defined, since we do not have a sigma to sum over
    pass
