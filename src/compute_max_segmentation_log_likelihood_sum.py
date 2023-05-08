"""This module computes the miniman segments term for the PMI scores with dynamic programming."""
import duckdb

from src.utils import get_ngram_table_name, get_token_field_name, Field


def get_join_condition(main_table_alias: str, sub_table_alias: str, sub_ngram_size: int, start_i: int) -> str:
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
    if sub_ngram_size == 1:
        return f'{sub_table_alias}.{Field.log_likelihood}'
    else:
        return f'greatest({sub_table_alias}.{Field.log_likelihood}, ' \
               f'{sub_table_alias}.{Field.max_segmentation_log_likelihood_sum})'


def get_update_query(ngram_size: int, split_i: int):
    # TODO: need to add aliases to the tables to avoid errors if we take from the same table.
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
        new_value_str = f'greatest({main_table_alias}.{Field.max_segmentation_log_likelihood_sum},' \
                        f' {new_value_str})'
    set_clause = f'SET {Field.max_segmentation_log_likelihood_sum} = {new_value_str}'

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
    # TODO: refactor
    #  test
    #  document
    # TODO: use logging for future scalability analysis. I want to access the
    # TODO: later, shorten / change the name of the tables...
    for ngram_size in range(2, max_ngram_size + 1):
        table_name = get_ngram_table_name(ngram_size)
        add_column_query = f"ALTER TABLE {table_name} ADD COLUMN " \
                           f"{Field.max_segmentation_log_likelihood_sum} DOUBLE;"
        db_connection.execute(add_column_query)

        for split_i in range(1, ngram_size):
            update_query = get_update_query(ngram_size, split_i)
            db_connection.execute(update_query)


def print_example_query():
    update_query = get_update_query(ngram_size=5, split_i=2)
    print(update_query)


if __name__ == '__main__':
    print_example_query()
