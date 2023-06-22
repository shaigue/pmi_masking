"""Module that extracts required information from the logs"""
# TODO: document
import datetime
import re
from collections.abc import Callable
from typing import Any

from src.utils import get_log_file, space_str, time_str

COMPUTE_STEPS = [
    'count_ngrams_in_batches',
    'aggregate_ngram_counts',
    'prune_low_count_ngrams',
    'compute_log_likelihood',
    'compute_max_segmentation_log_likelihood_sum',
    'compute_pmi_score',
    'compute_pmi_masking_vocab',
]


def parse_log_line(line: str) -> dict[str, str]:
    """Parse a single log line into a dictionary of the different parts of the log message"""
    datetime_regex = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
    module_name_regex = r'[\w.]+'
    level_regex = r'\w+'
    message_regex = r'.*'
    log_line_regex = f'(?P<datetime>{datetime_regex}),\\d+ - ' \
                     f'(?P<module_name>{module_name_regex}) - ' \
                     f'(?P<level>{level_regex}) - ' \
                     f'(?P<message>{message_regex})'
    log_line_regex = re.compile(log_line_regex)
    return log_line_regex.match(line).groupdict()


def read_log() -> list[str]:
    """Reads the log file and splits it into lines."""
    log_file = get_log_file()
    with log_file.open('r') as f:
        lines = f.read()
    lines = lines.splitlines(keepends=False)
    return lines


def read_and_parse_log() -> list[dict[str, str]]:
    """Read and parses the log lines into their parts"""
    lines = read_log()
    lines = [parse_log_line(line) for line in lines]
    return lines


def get_indices_that_satisfy_condition(lines: list, condition: Callable[[Any], bool]) -> list[int]:
    """Returns a list of indices that satisfy a given condition"""
    return [i for i, line in enumerate(lines) if condition(line)]


def slice_by_start_end_conditions(lines: list, start_condition: Callable[[Any], bool],
                                  end_condition: Callable[[Any], bool]) -> list[list]:
    """Slices a list by start and end conditions.
    Each slice is all the lines that appear between a start condition and the next end condition, inclusive.
    """
    start_indices = get_indices_that_satisfy_condition(lines, start_condition)
    end_indices = get_indices_that_satisfy_condition(lines, end_condition)
    if len(start_indices) < len(end_indices):
        raise RuntimeError(f'found {len(start_indices)} lines satisfying the start condition and {len(end_indices)} '
                           f'satisfying the end condition. there cannot be more starts than ends.')

    if len(start_indices) > len(end_indices):
        # there might have been starts that have not finished.
        # match each end with the closest start (the maximal start that comes before it.)
        new_start_indices = []
        for end_i in end_indices:
            matching_start_i = max(start_i for start_i in start_indices if start_i < end_i)
            new_start_indices.append(matching_start_i)
        start_indices = new_start_indices

    if any(start_i >= end_i for start_i, end_i in zip(start_indices, end_indices)):
        raise RuntimeError('found start condition after end condition.')

    return [lines[start_i:end_i+1] for start_i, end_i in zip(start_indices, end_indices)]


def get_regex_match_condition(regex: re.Pattern, field: str) -> Callable[[dict[str, str]], bool]:
    """Returns a function that returns True, iff a parsed log line field matches a regex."""
    def regex_match_condition(parsed_line: dict[str, str]):
        return regex.match(parsed_line[field]) is not None
    return regex_match_condition


def parse_matching_messages(parsed_lines: list[dict[str, str]], regex: re.Pattern) -> list[dict[str, str]]:
    """Applies a regex to the message field of log lines, only to those that match the regex"""
    match_objects = [regex.match(parsed_line['message']) for parsed_line in parsed_lines]
    return [match_object.groupdict() for match_object in match_objects if match_object is not None]


def get_parsed_line_datetime(parsed_line: dict[str, str]) -> datetime.datetime:
    """Converts the string datetime in the parsed log line into a datetime object"""
    datetime_format = '%Y-%m-%d %H:%M:%S'
    time_str = parsed_line['datetime']
    return datetime.datetime.strptime(time_str, datetime_format)


def get_parsed_lines_timediff_seconds(parsed_line1: dict[str, str], parsed_line2: dict[str, str]) -> int:
    """Returns the time difference between two parsed log lines, in seconds."""
    datetime1 = get_parsed_line_datetime(parsed_line1)
    datetime2 = get_parsed_line_datetime(parsed_line2)
    delta = datetime2 - datetime1
    return delta.seconds


def get_last_experiment_lines(lines: list[dict[str, str]], experiment_name: str) -> list[dict[str, str]]:
    """Returns the log lines of an experiment with a given name. If the experiment was run multiple times, the last
    time will be returned."""
    start_experiment_regex = re.compile(f'start experiment: {experiment_name}')
    start_experiment_condition = get_regex_match_condition(start_experiment_regex, 'message')
    end_experiment_regex = re.compile(f'end experiment: {experiment_name}')
    end_experiment_condition = get_regex_match_condition(end_experiment_regex, 'message')
    experiment_lines = slice_by_start_end_conditions(lines, start_experiment_condition,
                                                     end_experiment_condition)[-1]
    return experiment_lines


def extract_total_time(experiment_lines: list[dict[str, str]]) -> float:
    experiment_start_line = experiment_lines[0]
    experiment_end_line = experiment_lines[-1]
    return get_parsed_lines_timediff_seconds(experiment_start_line, experiment_end_line)


def extract_compute_steps_times(experiment_lines: list[dict[str, str]]) -> dict[str, float]:
    """Extracts the time each compute step took, in seconds."""
    compute_steps_times = {}
    for compute_step in COMPUTE_STEPS:
        module_name = f'src.db_implementation.{compute_step}'

        def start_condition(parsed_line: dict[str, str]) -> bool:
            return parsed_line['module_name'] == module_name and parsed_line['message'] == 'start'

        def end_condition(parsed_line: dict[str, str]) -> bool:
            return parsed_line['module_name'] == module_name and parsed_line['message'] == 'end'

        compute_step_parsed_lines = slice_by_start_end_conditions(experiment_lines, start_condition,
                                                                  end_condition)
        if len(compute_step_parsed_lines) != 1:
            raise RuntimeError

        start_line = compute_step_parsed_lines[0][0]
        end_line = compute_step_parsed_lines[0][-1]
        compute_steps_times[compute_step] = get_parsed_lines_timediff_seconds(start_line, end_line)

    return compute_steps_times


def extract_n_tokens(experiment_lines: list[dict[str, str]]) -> int:
    n_tokens_regex = re.compile(r'n_tokens: (?P<n_tokens>\d+)')
    parsed_n_tokens_messages = parse_matching_messages(experiment_lines, n_tokens_regex)
    if len(parsed_n_tokens_messages) != 1:
        raise RuntimeError
    return int(parsed_n_tokens_messages[0]['n_tokens'])


def extract_n_workers(experiment_lines: list[dict[str, str]]) -> int:
    regex = re.compile(r'n_workers: (?P<n_workers>\d+)')
    parsed_messages = parse_matching_messages(experiment_lines, regex)
    if len(parsed_messages) != 1:
        raise RuntimeError
    return int(parsed_messages[0]['n_workers'])


def extract_batch_files_total_size(experiment_lines: list[dict[str, str]]) -> float:
    batch_file_size_regex = re.compile(r'samples \d+-\d+, ngram size \d+, parquet file size: (?P<file_size>\d+)')
    parsed_batch_file_size_messages = parse_matching_messages(experiment_lines, batch_file_size_regex)
    if len(parsed_batch_file_size_messages) == 0:
        raise RuntimeError
    batch_file_sizes = [int(batch_info['file_size']) for batch_info in parsed_batch_file_size_messages]
    batch_files_total_size = sum(batch_file_sizes)
    return batch_files_total_size


def extract_db_size_after_aggregate_ngram_counts(experiment_lines: list[dict[str, str]]) -> int:
    regex = re.compile(r'db size bytes after aggregate counts: (?P<db_size>\d+)')
    parsed_lines = parse_matching_messages(experiment_lines, regex)
    if len(parsed_lines) != 1:
        raise RuntimeError
    return int(parsed_lines[0]['db_size'])


def extract_db_size_after_compute_pmi_score(experiment_lines: list[dict[str, str]]) -> int:
    regex = re.compile(r'db size bytes after pmi compute: (?P<db_size>\d+)')
    parsed_lines = parse_matching_messages(experiment_lines, regex)
    if len(parsed_lines) != 1:
        raise RuntimeError
    return int(parsed_lines[0]['db_size'])


def extract_os(experiment_lines: list[dict[str, str]]) -> str:
    regex = re.compile(r'os: (?P<os>.*)')
    parsed_messages = parse_matching_messages(experiment_lines, regex)
    if len(parsed_messages) != 1:
        raise RuntimeError
    return parsed_messages[0]['os']


def extract_ram(experiment_lines: list[dict[str, str]]) -> int:
    regex = re.compile(r'RAM_size: (?P<RAM_size>\d+)')
    parsed_messages = parse_matching_messages(experiment_lines, regex)
    if len(parsed_messages) != 1:
        raise RuntimeError
    return int(parsed_messages[0]['RAM_size'])


def extract_processor(experiment_lines: list[dict[str, str]]) -> str:
    regex = re.compile(r'processor: (?P<processor>.*)')
    parsed_messages = parse_matching_messages(experiment_lines, regex)
    if len(parsed_messages) != 1:
        raise RuntimeError
    return parsed_messages[0]['processor']


def extract_dataset_name(experiment_lines: list[dict[str, str]]) -> str:
    regex = re.compile(r'dataset: (?P<dataset_name>.*)')
    parsed_messages = parse_matching_messages(experiment_lines, regex)
    if len(parsed_messages) != 1:
        raise RuntimeError
    return parsed_messages[0]['dataset_name']


def extract_experiment_information_from_logs(experiment_name: str) -> dict:
    experiment_info = {}
    lines = read_and_parse_log()
    experiment_lines = get_last_experiment_lines(lines, experiment_name)

    experiment_info['OS'] = extract_os(experiment_lines)
    experiment_info['RAM_size'] = extract_ram(experiment_lines)
    experiment_info['processor'] = extract_processor(experiment_lines)
    experiment_info['total_time'] = extract_total_time(experiment_lines)
    experiment_info['dataset_name'] = extract_dataset_name(experiment_lines)
    experiment_info['compute_steps_times'] = extract_compute_steps_times(experiment_lines)
    experiment_info['n_tokens'] = extract_n_tokens(experiment_lines)
    experiment_info['n_workers'] = extract_n_workers(experiment_lines)
    experiment_info['batch_files_total_size'] = extract_batch_files_total_size(experiment_lines)
    experiment_info['db_size_after_aggregate_ngram_counts'] = \
        extract_db_size_after_aggregate_ngram_counts(experiment_lines)
    experiment_info['db_size_after_compute_pmi_score'] = \
        extract_db_size_after_compute_pmi_score(experiment_lines)
    experiment_info['total_space'] = max(
        experiment_info['db_size_after_compute_pmi_score'],
        experiment_info['db_size_after_aggregate_ngram_counts'],
        experiment_info['batch_files_total_size']
    )

    return experiment_info


def print_performance_result_line(experiment_info: dict):
    row = [
        experiment_info['dataset_name'],
        experiment_info['processor'],
        str(experiment_info['n_workers']),
        space_str(experiment_info['RAM_size']),
        experiment_info['OS'],
        time_str(experiment_info['total_time']),
        space_str(experiment_info['total_space'])
    ]
    row = '| ' + ' | '.join(row) + ' |'
    print(row)


if __name__ == '__main__':
    # res = extract_experiment_information_from_logs('end_to_end_test')
    res = extract_experiment_information_from_logs('bookcorpus_medium')
    # res = extract_experiment_information_from_logs('bookcorpus')
    print_performance_result_line(res)