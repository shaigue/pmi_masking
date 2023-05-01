"""Module for aggregating batch counts to a single database"""
import logging
from pathlib import Path

import duckdb

from src.utils import get_token_field_declaration_str, get_count_field_declaration_str, get_ngram_counts_table_name, \
    get_key_str, get_default_logging_config

logging.basicConfig(**get_default_logging_config(__file__))

# TODO: end-to-end test with counting ngrams module
# TODO: plan scalability analysis and add logging to support that
#   - extrapolate time for the entire dataset. to do so, we need to capture the start/end time for each file and
#   data about the file.
#   - extrapolate space for the entire dataset. for that we can record the number of entries in the table after each
#   merge.
# TODO: run on the entire sample
# TODO: scalability analysis


def get_create_table_query(ngram_size: int) -> str:
    """Generates a query that creates a new table with a column for each token in the ngram
    in its total count. A different table should be created for every ngram size.
    :param ngram_size: size of the ngrams in this table
    :returns: a string of SQL query to be executed.
    """
    fields_declarations = [get_token_field_declaration_str(token_i) for token_i in range(ngram_size)] + \
                          [get_count_field_declaration_str()]
    fields_declaration_str = ', '.join(fields_declarations)
    table_name = get_ngram_counts_table_name(ngram_size)
    create_table_query = f'CREATE OR REPLACE TABLE {table_name}(' \
                         f'{fields_declaration_str}, ' \
                         f'PRIMARY KEY({get_key_str(ngram_size)})' \
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
    key_str = get_key_str(ngram_size)
    insert_query = f'INSERT INTO {get_ngram_counts_table_name(ngram_size)} ' \
                   f'SELECT {key_str}, count FROM {table_to_insert} ' \
                   f'ON CONFLICT ({key_str}) DO UPDATE ' \
                   f'SET count = count + excluded.count;'
    return insert_query


def aggregate_batch_ngram_counts(save_dir: Path, max_ngram_size: int,
                                 database_file: Path) -> None:
    """Collects all the batch counts from `save_dir` and aggregates them to a single database.
    each ngram size gets a separate table.
    :param save_dir: the directory where the ngram counts where saved. expects this
        directory to have a subdirectory for every ngram size, e.g., subdirectory '4' for
        counts of ngrams of size 4.
    :param max_ngram_size: the maximum size of ngram to consider.
    :param database_file: a file to place the resulting database in.
    """
    connection = duckdb.connect(str(database_file))

    logging.info('aggregate batch ngram counts - start')
    for ngram_size in range(1, max_ngram_size + 1):
        logging.info(f'aggregating ngrams of size {ngram_size}')

        create_table_query = get_create_table_query(ngram_size)
        logging.info(f'creating new table. executing query:\n{create_table_query}')
        connection.execute(create_table_query)

        ngram_size_dir = save_dir / str(ngram_size)
        parquet_files = list(ngram_size_dir.glob('*.parquet'))
        for parquet_file in parquet_files:
            table_to_insert = connection.from_parquet(str(parquet_file))
            merge_and_add_query = get_merge_and_add_counts_query(
                ngram_size,
                'table_to_insert'
            )
            logging.info(f'merging file {parquet_file} into table - start. executing query:\n{merge_and_add_query}')
            connection.execute(merge_and_add_query)
            logging.info(f'merging file {parquet_file} into table - end')

            # record the size of the table
            table_size = len(connection.sql(f'SELECT * FROM {get_ngram_counts_table_name(ngram_size)}'))
            logging.info(f'table size: {table_size}')

    connection.close()
    logging.info('aggregate batch ngram counts - end')


def main():
    save_dir = Path('../data')
    max_ngram_size = 5
    database_file = save_dir / 'ngram_data.duckdb'
    aggregate_batch_ngram_counts(save_dir, max_ngram_size, database_file)


if __name__ == '__main__':
    main()
