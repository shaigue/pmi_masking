"""Module for aggregating batch counts to a single database"""
import logging
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

# TODO: extract logging configuration to a separate file
date_str = datetime.now().strftime('%d-%m-%y')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    filename=f'../logs/aggregate_batch_ngram_counts_{date_str}.log',
    filemode='w'
)

# TODO: plan scalability analysis and add logging to support that
#   - extrapolate time for the entire dataset. to do so, we need to capture the start/end time for each file and
#   data about the file.
#   - extrapolate space for the entire dataset. for that we can record the number of entries in the table after each
#   merge.
# TODO: unittests + refactoring
# TODO: run on the entire sample
# TODO: scalability analysis
# TODO: docstrings


@dataclass
class SQLField:
    field_name: str
    field_type: str


def get_create_table_query(new_table_name: str, key_fields: list[SQLField], data_fields: list[SQLField]) -> str:
    # TODO: docstring
    # TODO: consider adding default values to the data fields
    fields = key_fields + data_fields
    fields_declaration = ', '.join(f'{field.field_name} {field.field_type}' for field in fields)
    primary_key_declaration = ', '.join(field.field_name for field in key_fields)
    create_table_query = f'CREATE OR REPLACE TABLE {new_table_name}(' \
                         f'{fields_declaration}, ' \
                         f'PRIMARY KEY({primary_key_declaration})' \
                         f');'
    return create_table_query


def aggregate_batch_ngram_counts():
    save_dir = Path('../data')
    max_ngram_size = 5
    database_file = save_dir / 'ngram_data.duckdb'
    connection = duckdb.connect(str(database_file))

    logging.info('aggregate batch ngram counts - start')
    for ngram_size in range(1, max_ngram_size + 1):
        logging.info(f'aggregating ngrams of size {ngram_size}')

        # create the new table
        table_name = f'ngrams_of_size_{ngram_size}'
        token_fields = [SQLField(field_name=f'token_{token_i}', field_type='UINTEGER')
                        for token_i in range(1, ngram_size + 1)]
        data_fields = [
            SQLField(field_name='count', field_type='UINTEGER'),
            SQLField(field_name='log_likelihood', field_type='DOUBLE'),
            SQLField(field_name='max_seg', field_type='DOUBLE'),
            SQLField(field_name='pmi_score', field_type='DOUBLE')
        ]
        create_table_query = get_create_table_query(
            new_table_name=table_name,
            key_fields=token_fields,
            data_fields=data_fields,
        )
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
            logging.info(f'merging file {parquet_file} into table - start')
            logging.info(insert_query)
            connection.execute(insert_query)
            logging.info(f'merging file {parquet_file} into table - end')

            # record the size of the table
            table_size = len(connection.sql(f'SELECT * FROM {table_name}'))
            logging.info(f'table size: {table_size}')

    connection.close()
    logging.info('aggregate batch ngram counts - end')


if __name__ == '__main__':
    aggregate_batch_ngram_counts()
