import duckdb

from src.db_implementation import fields
from src.utils import get_ngram_table_name, get_module_logger

logger = get_module_logger(__name__)


def compute_pmi_score_for_ngram_size(db_connection: duckdb.DuckDBPyConnection, ngram_size: int) -> None:
    """Adds a pmi_score column to the ngram data table of a given size, and computes those values.
    :param db_connection: an open read/write connection to a duckdb database.
    :param ngram_size: update the values of the table containing data on ngrams of this size.
    """
    table_name = get_ngram_table_name(ngram_size)
    add_column_query = f"ALTER TABLE {table_name} ADD COLUMN {fields.PMI_SCORE} DOUBLE;"
    logger.info(f'executing query: {add_column_query}')
    db_connection.execute(add_column_query)

    update_pmi_score_query = f"UPDATE {table_name} SET {fields.PMI_SCORE} = " \
                             f"2 * {fields.LOG_LIKELIHOOD} - {fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM};"
    logger.info(f'executing query: {update_pmi_score_query}')
    db_connection.execute(update_pmi_score_query)


def compute_pmi_score(db_connection: duckdb.DuckDBPyConnection, max_ngram_size: int) -> None:
    """Goes over all the ngram tables in the DB and adds a pmi_score column and computes it's value.
    :param db_connection: an open read/write connection to a duckdb database.
    :param max_ngram_size: the maximal ngram size to consider.
    """
    logger.info('start')
    for ngram_size in range(2, max_ngram_size + 1):
        logger.info(f'computing pmi scores for ngrams of size {ngram_size}')
        compute_pmi_score_for_ngram_size(db_connection, ngram_size)
    logger.info('end')
