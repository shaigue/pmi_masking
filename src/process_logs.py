"""Module that extracts required information from the logs"""
import datetime
import json
import re
from collections.abc import Callable
from typing import Any

from src.utils import get_log_file


# TODO: Another thing that i might want to do is to first parse the values, and then do the processing.


def get_n_tokens_processed() -> int:
    with total_ngram_per_size_file.open('r') as f:
        total_ngram_per_size = json.load(f)
    return total_ngram_per_size['1']



def get_last_index(lines: list[str], condition: Callable[[str], bool]) -> int:
    return list(get_indices_that_satisfy_condition(lines, condition))[-1]


def get_last_slice(lines: list[str], start_condition: Callable[[str], bool]) -> list[str]:
    last_start_index = get_last_index(lines, start_condition)
    return lines[last_start_index:]


def slice_lines(lines: list[str], start_condition: Callable[[str], bool],
                start_line_to_key: Callable[[str], Any]) -> dict[Any, list[str]]:
    slice_dict = {}
    last_key = None

    for line in lines:
        if start_condition(line):
            last_key = start_line_to_key(line)
            slice_dict[last_key] = []

        if last_key is not None:
            slice_dict[last_key].append(line)

    return slice_dict


# ==== recreated version ====

def seconds_to_hours(seconds: float) -> float:
    return seconds / 3_600


def seconds_to_days(seconds: float) -> float:
    return seconds_to_hours(seconds) / 24


def read_log_lines() -> list[str]:
    log_file = get_log_file()
    with log_file.open('r') as f:
        lines = f.read()
    lines = lines.splitlines(keepends=False)
    return lines


def parse_log_line(line: str) -> dict[str, str]:
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


def parse_log_lines(log_lines: list[str]) -> list[dict[str, str]]:
    return [parse_log_line(line) for line in log_lines]


def get_indices_that_satisfy_condition(lines: list, condition: Callable[[Any], bool]) -> list[int]:
    return [i for i, line in enumerate(lines) if condition(line)]


def slice_by_start_end_condition(lines: list, start_condition: Callable[[Any], bool],
                                 end_condition: Callable[[Any], bool]) -> list[list]:
    start_indices = get_indices_that_satisfy_condition(lines, start_condition)
    end_indices = get_indices_that_satisfy_condition(lines, end_condition)
    if len(start_indices) != len(end_indices):
        raise RuntimeError(f'found {len(start_indices)} lines satisfying the start condition and {len(end_indices)} '
                           f'satisfying the end condition.')
    if any(start_i >= end_i for start_i, end_i in zip(start_indices, end_indices)):
        raise RuntimeError('found start condition after end condition.')

    return [lines[start_i:end_i+1] for start_i, end_i in zip(start_indices, end_indices)]


def get_regex_match_condition(regex: re.Pattern, field: str) -> Callable[[dict[str, str]], bool]:
    def regex_match_condition(parsed_line: dict[str, str]):
        return regex.match(parsed_line[field]) is not None
    return regex_match_condition


def parse_matching_messages(parsed_lines: list[dict[str, str]], regex: re.Pattern) -> list[dict[str, str]]:
    match_objects = [regex.match(parsed_line['message']) for parsed_line in parsed_lines]
    return [match_object.groupdict() for match_object in match_objects if match_object is not None]


def get_parsed_line_datetime(parsed_line: dict[str, str]) -> datetime.datetime:
    datetime_format = '%Y-%m-%d %H:%M:%S'
    time_str = parsed_line['datetime']
    return datetime.datetime.strptime(time_str, datetime_format)


def get_parsed_lines_timediff_seconds(parsed_line1: dict[str, str], parsed_line2: dict[str, str]) -> int:
    datetime1 = get_parsed_line_datetime(parsed_line1)
    datetime2 = get_parsed_line_datetime(parsed_line2)
    delta = datetime2 - datetime1
    return delta.seconds


def extract_experiment_information_from_logs(experiment_name: str) -> dict:
    log_lines = read_log_lines()
    parsed_log_lines = parse_log_lines(log_lines)

    start_experiment_regex = re.compile(f'start experiment_config: experiment_config.{experiment_name}')
    start_experiment_condition = get_regex_match_condition(start_experiment_regex, 'message')
    end_experiment_regex = re.compile(f'end experiment_config: experiment_config.{experiment_name}')
    end_experiment_condition = get_regex_match_condition(end_experiment_regex, 'message')
    experiment_lines = slice_by_start_end_condition(parsed_log_lines, start_experiment_condition,
                                                    end_experiment_condition)[-1]

    # find the total time. take the time of the last line and subtract the time of the first line
    total_time_seconds = get_parsed_lines_timediff_seconds(experiment_lines[0], experiment_lines[-1])

    # find the number of tokens processed
    n_tokens_regex = re.compile(r'n_tokens: (?P<n_tokens>\d+)')
    parsed_n_tokens_messages = parse_matching_messages(experiment_lines, n_tokens_regex)
    if len(parsed_n_tokens_messages) != 1:
        raise RuntimeError('there should only be one matching line')
    n_tokens = int(parsed_n_tokens_messages[0]['n_tokens'])

    # find the total size of the batch counts files
    batch_info_regex = re.compile(r'ngram_size: (?P<ngram_size>\d+), '
                                  r'ngrams_before_prune: (?P<ngrams_before_prune>\d+), '
                                  r'ngrams_after_prune: (?P<ngrams_after_prune>\d+), '
                                  r'batch_ngram_counts_file_size_bytes: (?P<batch_ngram_counts_file_size_bytes>\d+)')
    parsed_batch_info_messages = parse_matching_messages(experiment_lines, batch_info_regex)
    batch_ngram_counts_file_size_bytes_list = [int(batch_info['batch_ngram_counts_file_size_bytes'])
                                               for batch_info in parsed_batch_info_messages]
    total_batch_ngram_counter_files_size = sum(batch_ngram_counts_file_size_bytes_list)

    # find the line f'db_size_bytes: {db_size_bytes}' and save the value
    db_size_regex = re.compile(r'db_size_bytes: (?P<db_size_bytes>\d+)')
    db_size_lines = parse_matching_messages(experiment_lines, db_size_regex)
    if len(db_size_lines) != 1:
        raise RuntimeError(f'There are {len(db_size_lines)} matches. there should be exactly 1.')
    db_file_size_after_pmi_score_compute = int(db_size_lines[0]['db_size_bytes'])

    return {
        'n_tokens': n_tokens,
        'total_batch_ngram_counter_files_size': total_batch_ngram_counter_files_size,
        'db_file_size_after_pmi_score_compute': db_file_size_after_pmi_score_compute,
        'total_time_seconds': total_time_seconds
    }


if __name__ == '__main__':
    res = extract_experiment_information_from_logs('end_to_end_test')
    print(res)
