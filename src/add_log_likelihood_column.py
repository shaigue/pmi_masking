"""A module that adds a new log_likelihood column to the counts table"""
import math
from pathlib import Path

import duckdb

from src.utils import get_ngram_counts_table_name

# TODO: consider passing a connection instead of a file, this might reduce number of calls to open / close.
# TODO: test
# TODO: refactor
# TODO: document
# TODO: add to run_pipeline script, and also a main that performs this task.


def add_log_likelihood_column(database_file: Path, ngrams_of_size_count: dict[int, int]):
    connection = duckdb.connect(str(database_file))

    for ngram_size, ngram_count in ngrams_of_size_count.items():
        table_name = get_ngram_counts_table_name(ngram_size)
        add_col_to_table_query = f"ALTER TABLE {table_name} ADD COLUMN log_likelihood DOUBLE;"
        connection.execute(add_col_to_table_query)
        log_total_count = math.log(ngram_count)
        update_table_query = f"UPDATE {table_name} SET log_likelihood = ln(count) - {log_total_count};"
        connection.execute(update_table_query)

    connection.close()


def main():
    # TODO: run with the real database
    database_file = Path('../data/ngram_data.duckdb')


if __name__ == '__main__':
    main()
