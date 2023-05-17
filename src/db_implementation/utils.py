import json
from pathlib import Path

import duckdb


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
