"""This module computes the miniman segments term for the PMI scores with dynamic programming."""
import duckdb

from src.utils import get_ngram_table_name, get_token_field_name, get_module_logger
from src import fields


logger = get_module_logger(__name__)


def get_join_condition(main_table_alias: str, sub_table_alias: str, sub_ngram_size: int, start_i: int) -> str:
    """Generates a string representing the condition on the join between the main table and the sub ngram table.
    The main table is the ngram size that is being updated, and the sub table is the table that it's data is used
    to compute the new value of the main table.
    :param main_table_alias: the alias of the main table
    :param sub_table_alias: the alias of the sub table
    :param sub_ngram_size: the size of the sub ngram
    :param start_i: the index on which we start matching the keys of the main table and the sub table.
    :return: a string representing the join condition in SQL syntax.
    """
    join_fields = [
        (get_token_field_name(start_i + i), get_token_field_name(i))
        for i in range(sub_ngram_size)
    ]
    join_condition = ' AND '. join(
        f'{main_table_alias}.{field1} = {sub_table_alias}.{field2}'
        for field1, field2 in join_fields
    )
    return f'({join_condition})'


def get_sub_table_value_str(sub_table_alias: str, sub_ngram_size: int) -> str:
    """Returns a string in SQL syntax that represent the value for the sub-ngram that we want to propogate.
    If the ngram is of size 1, only log likelihood is available.
    If the ngram is of size greater than 1, we take the maximum of it's log likelihood and
    max segmentation log likelihood sum
    """
    if sub_ngram_size == 1:
        return f'{sub_table_alias}.{fields.LOG_LIKELIHOOD}'
    else:
        return f'greatest({sub_table_alias}.{fields.LOG_LIKELIHOOD}, ' \
               f'{sub_table_alias}.{fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM})'


def get_update_query(ngram_size: int, split_i: int) -> str:
    """Creates an SQL query that computes and updates the values for a given split point.
    :param ngram_size: the ngrams of this size will be updated by the query
    :param split_i: the index where the ngram is split
    :return: an UPDATE SQL query
    """
    left_sub_ngram_size = split_i
    right_sub_ngram_size = ngram_size - left_sub_ngram_size

    main_table_name = get_ngram_table_name(ngram_size)
    left_sub_table_name = get_ngram_table_name(left_sub_ngram_size)
    right_sub_table_name = get_ngram_table_name(right_sub_ngram_size)

    main_table_alias = 'main_table'
    left_table_alias = 'left_table'
    right_table_alias = 'right_table'

    left_value_str = get_sub_table_value_str(left_table_alias, left_sub_ngram_size)
    right_value_str = get_sub_table_value_str(right_table_alias, right_sub_ngram_size)
    # if it is the first update, just insert the new value. Otherwise, take the max
    new_value_str = f'{left_value_str} + {right_value_str}'
    if left_sub_ngram_size > 1:
        new_value_str = f'greatest({main_table_alias}.{fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM},' \
                        f' {new_value_str})'
    set_clause = f'SET {fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM} = {new_value_str}'

    # order the values according to the token ids
    left_join_condition = get_join_condition(main_table_alias, left_table_alias, left_sub_ngram_size, 0)
    right_join_condition = get_join_condition(main_table_alias, right_table_alias, right_sub_ngram_size, split_i)
    from_clause = f'FROM {main_table_name} {main_table_alias} ' \
                  f'INNER JOIN {left_sub_table_name} {left_table_alias} ON {left_join_condition} ' \
                  f'INNER JOIN {right_sub_table_name} {right_table_alias} ON {right_join_condition} '

    # make sure that insertions are aligned to the token ids
    where_condition = ' AND '.join(
        f'{main_table_alias}.{get_token_field_name(i)} = {main_table_name}.{get_token_field_name(i)}'
        for i in range(ngram_size)
    )
    where_clause = f'WHERE ({where_condition})'

    update_query = f'UPDATE {main_table_name} {set_clause} {from_clause} {where_clause}'
    return update_query


def compute_max_segmentation_log_likelihood_sum(db_connection: duckdb.DuckDBPyConnection, max_ngram_size: int) -> None:
    """Computes the maximal segmentation log likelihood sum for each ngram, to be used in the PMI score computation.
    Adds a new column to the table.
    :param db_connection: an open connection to the DB containing the ngram data tables.
    :param max_ngram_size: the maximal ngram size to consider.
    """
    # TODO: later, shorten / change the name of the tables...
    logger.info('start')
    for ngram_size in range(2, max_ngram_size + 1):
        logger.info(f'computing for ngram_size={ngram_size}')

        table_name = get_ngram_table_name(ngram_size)
        add_column_query = f"ALTER TABLE {table_name} ADD COLUMN " \
                           f"{fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM} DOUBLE;"
        db_connection.execute(add_column_query)

        for split_i in range(1, ngram_size):
            update_query = get_update_query(ngram_size, split_i)
            logger.info(f'doing update for split_i={split_i}')
            logger.info(f'executing query={update_query}')
            db_connection.execute(update_query)

    logger.info('end')
