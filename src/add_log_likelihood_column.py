"""A module that adds a new log_likelihood column to the counts table"""
import math
from pathlib import Path

import duckdb

from src.utils import get_ngram_counts_table_name, read_total_ngrams_per_size


# TODO: consider passing a connection instead of a file, this might reduce number of calls to open / close.
# TODO: run on the database and prepare for the next step.


def add_log_likelihood_column(database_file: Path, total_ngrams_per_size: dict[int, int]) -> None:
    """Adds a log_likelihood column to the database containing the keys (token ids) and the counts.
    :param database_file: the file that the database is hosted on.
    :param total_ngrams_per_size: a dictionary containing the number of times ngrams of a given size appear in the
        dataset
    """
    connection = duckdb.connect(str(database_file))

    for ngram_size, ngram_count in total_ngrams_per_size.items():
        table_name = get_ngram_counts_table_name(ngram_size)
        add_col_to_table_query = f"ALTER TABLE {table_name} ADD COLUMN log_likelihood DOUBLE;"
        connection.execute(add_col_to_table_query)
        log_total_count = math.log(ngram_count)
        update_table_query = f"UPDATE {table_name} SET log_likelihood = ln(count) - {log_total_count};"
        connection.execute(update_table_query)

    connection.close()


def main():
    save_dir = Path('../data')
    database_file = save_dir / 'ngram_data.duckdb'
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    add_log_likelihood_column(database_file, total_ngrams_per_size)


if __name__ == '__main__':
    main()
