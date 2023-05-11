import json
import logging
from math import floor
from pathlib import Path

import duckdb
from datasets import Dataset as HuggingfaceDataset
from transformers import PreTrainedTokenizerBase


Ngram = tuple[int, ...]


def get_module_logger(name: str) -> logging.Logger:
    """
    :param name: should be __name__ special variable in the module that calls this
        function.
    """
    # TODO: maybe add some config file to the directory for project
    #  level configurations.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logs_dir = Path(__file__).parents[1] / 'logs'
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / 'log.log'
    file_handler = logging.FileHandler(str(log_file), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_token_field_name(token_i: int) -> str:
    return f'token_{token_i}'


def get_token_field_sql_type() -> str:
    return 'UINTEGER'


def get_count_field_sql_type() -> str:
    return 'UINTEGER'


def get_token_field_declaration_str(token_i: int) -> str:
    return f'{get_token_field_name(token_i)} {get_token_field_sql_type()}'


def get_count_field_declaration_str() -> str:
    return f'count {get_count_field_sql_type()}'


def get_ngram_table_name(ngram_size: int) -> str:
    return f'ngram_of_size_{ngram_size}_table'


def get_token_field_names(ngram_size: int) -> list[str]:
    return [get_token_field_name(token_i) for token_i in range(ngram_size)]


def get_token_fields_str(ngram_size: int) -> str:
    return ', '.join(get_token_field_name(token_i) for token_i in range(ngram_size))


def get_total_ngrams_per_size_file(save_dir):
    return save_dir / 'total_ngrams_per_size.json'


def read_total_ngrams_per_size(save_dir: Path) -> dict[int, int]:
    json_file = get_total_ngrams_per_size_file(save_dir)
    with json_file.open('r') as f:
        total_ngrams_per_size = json.load(f)

    total_ngrams_per_size = {int(key): value for key, value in total_ngrams_per_size.items()}
    return total_ngrams_per_size


def tokenize_dataset(dataset: HuggingfaceDataset, tokenizer: PreTrainedTokenizerBase,
                     n_workers: int, tokenizer_batch_size: int):
    def tokenize(batch: dict[str, list]):
        return tokenizer(batch['text'], add_special_tokens=False)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=n_workers,
    )
    return dataset


def validate_ngram_size_to_vocab_percent(ngram_size_to_vocab_percent: dict[int, float]):
    if 1 in ngram_size_to_vocab_percent.keys():
        raise ValueError('there should not be ngrams of size 1 in the vocabulary. '
                         f'input suggests that there should be {ngram_size_to_vocab_percent[1]}% ngrams of size 1.')

    total_percent = sum(ngram_size_to_vocab_percent.values())
    if total_percent != 100:
        raise ValueError(f'the total percents should sum up to 100. '
                         f'input: {ngram_size_to_vocab_percent} sums to {total_percent}.')


def compute_number_of_ngrams_per_size_in_vocab(ngram_size_to_vocab_percent: dict[int, float], vocab_size: int):
    # compute how much ngrams per size we take
    number_of_ngrams_of_size_in_vocab = {}
    for ngram_size, vocab_percent in ngram_size_to_vocab_percent.items():
        number_of_ngrams_of_size_in_vocab[ngram_size] = floor(vocab_size * vocab_percent / 100)
    # take the extra tokens from the smallest ngram_size (=2)
    extra_ngrams = vocab_size - sum(number_of_ngrams_of_size_in_vocab.values())
    number_of_ngrams_of_size_in_vocab[2] += extra_ngrams
    return number_of_ngrams_of_size_in_vocab


def open_db_connection(save_dir: Path):
    database_file = save_dir / 'ngram_data.duckdb'
    db_connection = duckdb.connect(str(database_file))
    return db_connection
