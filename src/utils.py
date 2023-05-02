# TODO: move here functions that are used in multiple modules. I don't want
#   the different modules to be inter-dependent
import logging
from datetime import datetime
from pathlib import Path


def get_default_logging_config(file: str) -> dict:
    """
    :param file: should be __file__ special variable in the module that calls this
        function.
    """
    # TODO: maybe add some config file to the directory for project
    #  level configurations.
    logs_dir = Path(__file__).parents[1] / 'logs'
    date_str = datetime.now().strftime('%d-%m-%y')
    filename = Path(file).stem
    filename = f'{filename}_{date_str}.log'
    log_file = logs_dir / filename
    return dict(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            filename=str(log_file),
            filemode='a'
    )


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


def get_ngram_counts_table_name(ngram_size: int) -> str:
    return f'ngram_of_size_{ngram_size}_counts_table'


def get_token_field_names(ngram_size: int) -> list[str]:
    return [get_token_field_name(token_i) for token_i in range(ngram_size)]


def get_key_str(ngram_size: int) -> str:
    return ', '.join(get_token_field_name(token_i) for token_i in range(ngram_size))
