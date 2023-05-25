from collections.abc import Callable
from math import log2
from pprint import pprint

import n_tokens
from src.process_logs import extract_experiment_information_from_logs
from src.utils import time_str, space_str
# TODO: should I document this? it is for internal use...

def extrapolate_linear(size1: int, size2: int, value1: float) -> float:
    return (size2 / size1) * value1


def extrapolate_n_log_n(size1: int, size2: int, value1: float) -> float:
    n_log_n1 = size1 * log2(size1)
    n_log_n2 = size2 * log2(size2)
    return (n_log_n2 / n_log_n1) * value1


def multiply_func(func: Callable, constant: float):
    def new_func(*args):
        return constant * func(*args)

    return new_func


def extrapolate_linear_times_sqrt(size1: float, size2: float, value1: float) -> float:
    x1 = pow(size1, 1.5)
    x2 = pow(size2, 1.5)
    value2 = (x2 / x1) * value1
    return value2


# COMPUTE_STEPS_EXTRAPOLATION_METHODS = {
#     'count_ngrams_in_batches': extrapolate_linear,
#     'aggregate_ngram_counts': extrapolate_n_log_n,
#     'prune_low_count_ngrams': extrapolate_n_log_n,
#     'compute_log_likelihood': extrapolate_linear,
#     'compute_max_segmentation_log_likelihood_sum': extrapolate_n_log_n,
#     'compute_pmi_score': extrapolate_linear,
#     'compute_pmi_masking_vocab': extrapolate_n_log_n,
# }

# COMPUTE_STEPS_EXTRAPOLATION_METHODS = {
#     'count_ngrams_in_batches': extrapolate_n_log_n,
#     'aggregate_ngram_counts': extrapolate_linear_times_sqrt,
#     'prune_low_count_ngrams': extrapolate_linear_times_sqrt,
#     'compute_log_likelihood': extrapolate_linear_times_sqrt,
#     'compute_max_segmentation_log_likelihood_sum': extrapolate_linear_times_sqrt,
#     'compute_pmi_score': extrapolate_linear_times_sqrt,
#     'compute_pmi_masking_vocab': extrapolate_linear_times_sqrt,
# }
# I use this functions to estimate since they are over-estimates, and I choose them according to a single experiment.
# I don't try to be as accurate as I can, but to give an appropriate overestimation.
COMPUTE_STEPS_EXTRAPOLATION_METHODS = {
    'count_ngrams_in_batches': extrapolate_linear,
    'aggregate_ngram_counts': multiply_func(extrapolate_n_log_n, 1.5),
    'prune_low_count_ngrams': multiply_func(extrapolate_n_log_n, 1.25),
    'compute_log_likelihood': multiply_func(extrapolate_linear, 1.25),
    'compute_max_segmentation_log_likelihood_sum': multiply_func(extrapolate_n_log_n, 2),
    'compute_pmi_score': multiply_func(extrapolate_linear, 4),
    'compute_pmi_masking_vocab': multiply_func(extrapolate_n_log_n, 2),
}


def estimate_total_space(n_tokens_target: int, experiment_info: dict) -> float:
    # assumption: total space increases linearly
    # this is not exact since the number of ngrams does not increase linearly in the number of tokens.
    return extrapolate_linear(experiment_info['n_tokens'], n_tokens_target, experiment_info['total_space'])


def estimate_total_time1(n_tokens_target: int, experiment_info: dict) -> float:
    # assumes total time increase n log (n) in the number of tokens.
    return extrapolate_n_log_n(experiment_info['n_tokens'], n_tokens_target, experiment_info['total_time'])


def estimate_total_time2(n_tokens_target: int, experiment_info: dict) -> tuple[float, dict[str, float]]:
    # split the times by each module. for each module, use the correct extrapolation method.
    # extrapolate using n_tokens
    time_per_step_estimate = {
        step: COMPUTE_STEPS_EXTRAPOLATION_METHODS[step](experiment_info['n_tokens'], n_tokens_target, time)
        for step, time in experiment_info['compute_steps_times'].items()
    }
    total_time_estimate = sum(time_per_step_estimate.values())
    return total_time_estimate, time_per_step_estimate


def print_experiment_resources_used(experiment_name: str) -> None:
    print(f'\n***actual resources used in {experiment_name}***\n')
    experiment_info = extract_experiment_information_from_logs(experiment_name)
    print(f"{experiment_info['n_tokens']:,} tokens processed in {experiment_name}")
    print(f"space for {experiment_name}: {space_str(experiment_info['total_space'])}")
    print(f"total time for {experiment_name}: {time_str(experiment_info['total_time'])}")
    print('per computation step:')
    pprint(experiment_info['compute_steps_times'])


def estimate_dataset_resources(experiment_name: str, target_dataset_name: str, target_n_tokens: int,
                               n_workers: int = None):
    print(f'\n***estimated resources for {target_dataset_name} based on {experiment_name} '
          f'with {n_workers} workers***\n')
    experiment_info = extract_experiment_information_from_logs(experiment_name)
    total_space_estimate = estimate_total_space(target_n_tokens, experiment_info)

    total_time_estimate1 = estimate_total_time1(target_n_tokens, experiment_info)
    total_time_estimate2, time_per_step_estimate = estimate_total_time2(target_n_tokens, experiment_info)

    time_factor = 1 if n_workers is None else (experiment_info['n_workers'] / n_workers)
    total_time_estimate1 = time_factor * total_time_estimate1
    total_time_estimate2 = time_factor * total_time_estimate2
    time_per_step_estimate = {k: time_factor * v for k, v in time_per_step_estimate.items()}

    print(f'space estimate for {target_dataset_name}: {space_str(total_space_estimate)}')
    print(f'estimated total time 1 for {target_dataset_name}: {time_str(total_time_estimate1)}')
    print(f'estimated total time 2 for {target_dataset_name}: {time_str(total_time_estimate2)}')
    print('estimated per computation step:')
    pprint(time_per_step_estimate)


if __name__ == '__main__':
    print_experiment_resources_used('medium_size_bookcorpus')
    print_experiment_resources_used('bookcorpus')
    estimate_dataset_resources('medium_size_bookcorpus', 'bookcorpus', n_tokens.BOOKCORPUS)
    estimate_dataset_resources('medium_size_bookcorpus', 'wikipedia', n_tokens.WIKIPEDIA, 5)
    estimate_dataset_resources('medium_size_bookcorpus', 'RedPajama', n_tokens.RED_PAJAMA, 5)
    estimate_dataset_resources('medium_size_bookcorpus', 'RedPajama', n_tokens.RED_PAJAMA, 10)
    estimate_dataset_resources('medium_size_bookcorpus', 'RedPajama', n_tokens.RED_PAJAMA, 15)
