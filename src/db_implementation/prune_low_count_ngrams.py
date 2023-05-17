import duckdb

from src.db_implementation import fields
from src.utils import get_module_logger
from src.db_implementation.utils import get_ngram_table_name

logger = get_module_logger(__name__)


def prune_low_count_ngrams(db_connection: duckdb.DuckDBPyConnection, max_ngram_size: int,
                           min_count_threshold: int) -> None:
    """Prunes ngrams that have counts lower than `min_count_threshold`."""
    logger.info('start')
    for ngram_size in range(1, max_ngram_size + 1):
        table_name = get_ngram_table_name(ngram_size)
        delete_query = f'DELETE FROM {table_name} WHERE {fields.COUNT} < {min_count_threshold}'
        db_connection.execute(delete_query)
    logger.info('end')
