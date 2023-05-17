import duckdb

from src import fields
from src.utils import get_ngram_table_name, get_module_logger

logger = get_module_logger(__name__)


def filter_low_count_ngrams(max_ngram_size: int, db_connection: duckdb.DuckDBPyConnection,
                            min_count_threshold: int) -> None:
    logger.info('start')
    for ngram_size in range(1, max_ngram_size + 1):
        table_name = get_ngram_table_name(ngram_size)
        delete_query = f'DELETE FROM {table_name} WHERE {fields.COUNT} < {min_count_threshold}'
        db_connection.execute(delete_query)
    logger.info('end')
