"""A module that adds a new log_likelihood column to the counts table"""
import math

import duckdb

from src.db_implementation import fields
from src.utils import get_ngram_table_name, get_module_logger

logger = get_module_logger(__name__)


def compute_log_likelihood(db_connection: duckdb.DuckDBPyConnection, total_ngrams_per_size: dict[int, int]) -> None:
    """Adds a log_likelihood column to the database containing the keys (token ids) and the counts.
    :param db_connection: an open read/write connection to a duckdb.
    :param total_ngrams_per_size: a dictionary containing the number of times ngrams of a given size appear in the
        dataset
    """
    logger.info('start')
    for ngram_size, ngram_count in total_ngrams_per_size.items():
        table_name = get_ngram_table_name(ngram_size)
        add_col_to_table_query = f"ALTER TABLE {table_name} ADD COLUMN {fields.LOG_LIKELIHOOD} DOUBLE;"
        db_connection.execute(add_col_to_table_query)
        log_total_count = math.log(ngram_count)
        update_table_query = f"UPDATE {table_name} SET {fields.LOG_LIKELIHOOD} = " \
                             f"ln({fields.COUNT}) - {log_total_count};"
        db_connection.execute(update_table_query)
    logger.info('end')
