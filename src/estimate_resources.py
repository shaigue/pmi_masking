# TODO: save the different versions of estimation, so we can track which one performs best.
from math import log2

import n_tokens
from src.process_logs import extract_experiment_information_from_logs


def extrapolate_linear(size1: int, size2: int, value1: float) -> float:
    return (size2 / size1) * value1


def extrapolate_n_log_n(size1: int, size2: int, value1: float) -> float:
    n_log_n1 = size1 * log2(size1)
    n_log_n2 = size2 * log2(size2)
    return (n_log_n2 / n_log_n1) * value1


def estimate_total_space_bytes(n_tokens_target: int, n_tokens_given: int, total_space_bytes_given: int) -> float:
    # assumption: total space increases linearly
    # this is not exact since the number of ngrams does not increase linearly in the number of tokens.
    # TODO: when estimating the space, also consider the size of the DB after the count aggregation step, before
    #    pruning. it might be larger than the final DB.
    return extrapolate_linear(n_tokens_given, n_tokens_target, total_space_bytes_given)


def total_space_from_experiment_info(experiment_info: dict) -> int:
    return max(
        experiment_info['total_batch_ngram_counter_files_size'],
        experiment_info['db_file_size_after_pmi_score_compute']
    )


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


def estimate_total_time_seconds(n_tokens_target: int, n_tokens_given: int, total_time_seconds_given: int) -> float:
    # assumes total time increase n log (n) in the number of tokens.
    return extrapolate_n_log_n(n_tokens_given, n_tokens_target, total_time_seconds_given)


def print_resources_used(experiment_name: str) -> None:
    experiment_info = extract_experiment_information_from_logs(experiment_name)
    n_tokens_experiment = experiment_info['n_tokens']
    print(f'{n_tokens_experiment:,} tokens processed in {experiment_name}')
    total_space_bytes_experiment = total_space_from_experiment_info(experiment_info)
    print(f'space for {experiment_name}: {space_str(total_space_bytes_experiment)}')
    total_time_seconds_experiment = experiment_info['total_time_seconds']
    print(f'total time for {experiment_name}: {time_str(total_time_seconds_experiment)}')


def estimate_dataset_resources(experiment_name: str, target_dataset_name: str, target_n_tokens: int):
    experiment_info = extract_experiment_information_from_logs(experiment_name)
    n_tokens_experiment = experiment_info['n_tokens']
    print(f'{n_tokens_experiment:,} tokens processed in {experiment_name}')
    total_space_bytes_experiment = total_space_from_experiment_info(experiment_info)
    print(f'space for {experiment_name}: {space_str(total_space_bytes_experiment)}')
    total_time_seconds_experiment = experiment_info['total_time_seconds']
    print(f'total time for {experiment_name}: {time_str(total_time_seconds_experiment)}')

    total_space_bytes_target = estimate_total_space_bytes(target_n_tokens, n_tokens_experiment,
                                                          total_space_bytes_experiment)
    print(f'space estimate for {target_dataset_name}: {space_str(total_space_bytes_target)}')
    total_time_seconds_target = estimate_total_time_seconds(target_n_tokens, n_tokens_experiment,
                                                            total_time_seconds_experiment)
    print(f'estimated total time for {target_dataset_name}: {time_str(total_time_seconds_target)}')


def estimate_bookcorpus():
    experiment_name = 'medium_size_bookcorpus'
    target_dataset_name = 'bookcorpus'
    estimate_dataset_resources(experiment_name, target_dataset_name, n_tokens.BOOKCORPUS)


def estimate_wikipedia():
    experiment_name = 'medium_size_bookcorpus'
    target_dataset_name = 'wikipedia'
    estimate_dataset_resources(experiment_name, target_dataset_name, n_tokens.WIKIPEDIA)


def estimate_red_pajama():
    experiment_name = 'medium_size_bookcorpus'
    target_dataset_name = 'red_pajama'
    estimate_dataset_resources(experiment_name, target_dataset_name, n_tokens.RED_PAJAMA)


if __name__ == '__main__':
    estimate_bookcorpus()
    print_resources_used('bookcorpus')
    # estimate_wikipedia()
    # estimate_red_pajama()
