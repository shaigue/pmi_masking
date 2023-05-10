# experiment parameters
import datetime
from collections.abc import Callable, Iterable
import math
from pathlib import Path
import json
import re
from typing import Any

from config import PROJECT_ROOT_PATH

# TODO: maybe use a configuration file for the 'run_pipline' file, and use it both for running and scalability analysis
ngram_count_batch_size = 1_000_000
n_samples = 30_000_000
n_workers = 3
max_ngram_size = 5
filter_ngram_count_threshold = 2
save_dir = PROJECT_ROOT_PATH / 'data'
log_file = PROJECT_ROOT_PATH / 'logs/log.log'
total_ngram_per_size_file = save_dir / 'total_ngrams_per_size.json'
N_TOKENS_WIKIPEDIA = 24_000_000_000         # 24 Billion
N_TOKENS_RED_PAJAMA = 1_200_000_000_000     # 1.2 Trillion


datetime_regex = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
datetime_format = '%Y-%m-%d %H:%M:%S'
module_name_regex = r'\w+\.\w+'
message_regex = r'.*'
log_line_regex = f'(?P<datetime>{datetime_regex}),\\d+ - ' \
                 f'(?P<module_name>{module_name_regex}) - \\w+ - ' \
                 f'(?P<message>{message_regex})'
log_line_regex = re.compile(log_line_regex)


def read_log_lines() -> list[str]:
    with log_file.open('r') as f:
        lines = f.read()
    lines = lines.splitlines(keepends=False)
    return lines


def get_n_tokens_processed() -> int:
    with total_ngram_per_size_file.open('r') as f:
        total_ngram_per_size = json.load(f)
    return total_ngram_per_size['1']

# TODO: Another thing that i might want to do is to first parse the values, and then do the processing.


def get_regex_match_condition(regex: re.Pattern) -> Callable[[str], bool]:
    def regex_match_condition(line: str) -> bool:
        return regex.match(line) is not None
    return regex_match_condition


def get_module_name_condition(module_name: str) -> Callable[[str], bool]:
    def module_name_condition(line: str) -> bool:
        return log_line_regex.match(line).group('module_name') == module_name
    return module_name_condition


def get_message_condition(message: str) -> Callable[[str], bool]:
    def message_condition(line: str) -> bool:
        return log_line_regex.match(line).group('message') == message
    return message_condition


def iter_line_indices(lines: list[str], condition: Callable[[str], bool]) -> Iterable[int]:
    for i, line in enumerate(lines):
        if condition(line):
            yield i


def get_last_index(lines: list[str], condition: Callable[[str], bool]) -> int:
    return list(iter_line_indices(lines, condition))[-1]


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


def get_line_datetime(line: str) -> datetime.datetime:
    time_str = log_line_regex.match(line).group('datetime')
    return datetime.datetime.strptime(time_str, datetime_format)


def get_lines_timedelta_seconds(line1: str, line2: str) -> int:
    return (get_line_datetime(line2) - get_line_datetime(line1)).seconds


def extrapolate_time_n_log_n(actual_n: int, expected_n: int, actual_time: int):
    constant = actual_time / (actual_n * math.log2(actual_n))
    return constant * (expected_n * math.log2(expected_n))


def seconds_to_hours(seconds: int) -> float:
    return seconds / 3_600


def seconds_to_days(seconds: int) -> float:
    return seconds_to_hours(seconds) / 24


def parse_line(line: str) -> dict[str, str]:
    match_object = log_line_regex.match(line)
    return match_object.groupdict()
