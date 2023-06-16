import json
from pathlib import Path

import duckdb

import config
from src.db_implementation.fields import get_token_field_name


def get_ngram_table_name(ngram_size: int) -> str:
    return f'ngram_of_size_{ngram_size}_table'


def get_token_field_names(ngram_size: int) -> list[str]:
    return [get_token_field_name(token_i) for token_i in range(ngram_size)]


def get_token_fields_str(ngram_size: int) -> str:
    return ', '.join(get_token_field_name(token_i) for token_i in range(ngram_size))


def get_total_ngrams_per_size_file(save_dir: Path):
    return save_dir / 'total_ngrams_per_size.json'


def read_total_ngrams_per_size(save_dir: Path) -> dict[int, int]:
    json_file = get_total_ngrams_per_size_file(save_dir)
    with json_file.open('r') as f:
        total_ngrams_per_size = json.load(f)

    total_ngrams_per_size = {int(key): value for key, value in total_ngrams_per_size.items()}
    return total_ngrams_per_size


def get_db_path(save_dir: Path) -> Path:
    return save_dir / 'ngram_data.duckdb'


def open_db_connection(save_dir: Path):
    database_file = get_db_path(save_dir)
    db_connection = duckdb.connect(str(database_file))
    return db_connection


def get_save_dir(experiment_name: str):
    return config.DATA_DIR / experiment_name


def get_vocab_file(experiment_name: str) -> Path:
    return config.VOCABS_DIR / f'{experiment_name}.txt'
