"""Module for aggregating batch counts to a single database"""
from pathlib import Path

import duckdb

from src.db_implementation import fields
from src.utils import get_module_logger
from src.db_implementation.utils import get_token_field_declaration_str, get_count_field_declaration_str, \
    get_ngram_table_name, get_token_fields_str

logger = get_module_logger(__name__)


def get_create_table_query(ngram_size: int) -> str:
    """Generates a query that creates a new table with a column for each token in the ngram
    in its total count. A different table should be created for every ngram size.
    :param ngram_size: size of the ngrams in this table
    :returns: a string of SQL query to be executed.
    """
    fields_declarations = [get_token_field_declaration_str(token_i) for token_i in range(ngram_size)] + \
                          [get_count_field_declaration_str()]
    fields_declaration_str = ', '.join(fields_declarations)
    table_name = get_ngram_table_name(ngram_size)
    create_table_query = f'CREATE OR REPLACE TABLE {table_name}(' \
                         f'{fields_declaration_str}, ' \
                         f'PRIMARY KEY({get_token_fields_str(ngram_size)})' \
                         f');'
    return create_table_query


def get_merge_and_add_counts_query(ngram_size: int, table_to_insert: str) -> str:
    """Generates an SQL query that inserts the table named `table_to_insert` that contains ngram counts into
    the table with the aggregated counts of the given ngram size. If an ngram is already present in the table,
    it adds the two count values.
    :param ngram_size: the size of the ngrams.
    :param table_to_insert: name of the table to insert
    :returns: an SQL query to execute
    """
    key_str = get_token_fields_str(ngram_size)
    insert_query = f'INSERT INTO {get_ngram_table_name(ngram_size)} ' \
                   f'SELECT {key_str}, {fields.COUNT} FROM {table_to_insert} ' \
                   f'ON CONFLICT ({key_str}) DO UPDATE ' \
                   f'SET {fields.COUNT} = {fields.COUNT} + excluded.{fields.COUNT};'
    return insert_query


def aggregate_ngram_counts(save_dir: Path, max_ngram_size: int,
                           db_connection: duckdb.DuckDBPyConnection) -> None:
    """Collects all the batch counts from `save_dir` and aggregates them to a single database.
    each ngram size gets a separate table.
    :param save_dir: the directory where the ngram counts where saved. expects this
        directory to have a subdirectory for every ngram size, e.g., subdirectory '4' for
        counts of ngrams of size 4.
    :param max_ngram_size: the maximum size of ngram to consider.
    :param db_connection: a read / write connection to a duckdb database.
    """
    logger.info('start')
    for ngram_size in range(1, max_ngram_size + 1):
        logger.info(f'aggregating ngrams of size {ngram_size}')

        create_table_query = get_create_table_query(ngram_size)
        logger.info(f'creating new table. executing query: {create_table_query}')
        db_connection.execute(create_table_query)

        ngram_size_dir = save_dir / str(ngram_size)
        parquet_files = list(ngram_size_dir.glob('*.parquet'))
        for file_i, parquet_file in enumerate(parquet_files, 1):
            logger.info(f'processing file {file_i} out of {len(parquet_files)}')
            table_to_insert = db_connection.from_parquet(str(parquet_file))
            merge_and_add_query = get_merge_and_add_counts_query(
                ngram_size,
                'table_to_insert'
            )
            logger.info(f'merging file {parquet_file} into table - start. executing query: {merge_and_add_query}')
            db_connection.execute(merge_and_add_query)
            logger.info(f'merging file {parquet_file} into table - end')
            parquet_file.unlink()

            # record the size of the table
            table_size = len(db_connection.sql(f'SELECT * FROM {get_ngram_table_name(ngram_size)}'))
            logger.info(f'table size: {table_size}')

        ngram_size_dir.rmdir()

    logger.info('end')
