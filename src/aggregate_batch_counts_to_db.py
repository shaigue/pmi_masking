"""Module for aggregating batch counts to a single database"""
import logging
from pathlib import Path

import duckdb


logging.basicConfig(level=logging.INFO)

# TODO: write code
# TODO: run on subsample
# TODO: plan scalability analysis and add logging to support that
# TODO: unittests + refactoring
# TODO: run on the entire sample
# TODO: scalability analysis
# TODO: docstrings


def aggregate_batch_counts_to_db():
    save_dir = Path('../data')
    max_ngram_size = 5
    database_file = save_dir / 'ngram_data.duckdb'
    connection = duckdb.connect(str(database_file))

    logging.info('aggregate ngram counts start')
    for ngram_size in range(1, max_ngram_size + 1):
        logging.info(f'aggregating ngrams of size {ngram_size}')

        # TODO: i think that the table creation can be extracted into a function and tested
        # create the table
        table_name = f'ngrams_of_size_{ngram_size}'
        token_field_names = [f'token_{token_i}' for token_i in range(1, ngram_size + 1)]
        field_names_and_types = [f'{token_field_name} UINTEGER' for token_field_name in token_field_names] + \
            ['count UINTEGER', 'log_likelihood DOUBLE', 'max_seg DOUBLE', 'pmi_score DOUBLE']

        # TODO: maybe add default values?
        fields_declaration = ', '.join(field_names_and_types)
        primary_key_declaration = ', '.join(token_field_names)
        create_table_query = f'CREATE OR REPLACE TABLE {table_name}(' \
                             f'{fields_declaration}, ' \
                             f'PRIMARY KEY({primary_key_declaration})' \
                             f');'
        logging.info(f'creating new table with query \n{create_table_query}')
        connection.execute(create_table_query)

        # insert the new data into the table without duplication
        ngram_size_dir = save_dir / str(ngram_size)
        parquet_files = list(ngram_size_dir.glob('*.parquet'))
        for parquet_file in parquet_files:
            table_to_insert = connection.from_parquet(str(parquet_file))
            insert_query = f'INSERT INTO {table_name} ' \
                           f'SELECT {primary_key_declaration}, count, 0, 0, 0 FROM table_to_insert ' \
                           f'ON CONFLICT ({primary_key_declaration}) DO UPDATE ' \
                           f'SET count = count + excluded.count;'
            logging.info(f'inserting parquet file {parquet_file} into table with query \n{insert_query}')
            connection.execute(insert_query)

    connection.close()
    logging.info('aggregate ngram counts end')


if __name__ == '__main__':
    aggregate_batch_counts_to_db()
