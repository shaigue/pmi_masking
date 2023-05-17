import math
import shutil
import unittest
from pathlib import Path

import duckdb

from src.db_implementation import fields
from src.db_implementation.compute_log_likelihood import compute_log_likelihood
from src.db_implementation.aggregate_ngram_counts import get_create_table_query
from src.db_implementation.utils import get_ngram_table_name


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path('../test_temp')
        self.temp_dir.mkdir(exist_ok=True)
        self.db_connection = duckdb.connect(':memory:')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)
        self.db_connection.close()

    def test_compute_log_likelihood(self):
        ngrams_of_size_count = {1: 9, 2: 8}
        ngram_counts_per_size = {
            1: {
                (1, ): 5,
                (2, ): 4,
            },
            2: {
                (1, 1): 3,
                (1, 2): 2,
                (2, 1): 1,
                (2, 2): 2
            }
        }
        # Create dummy tables
        for ngram_size, ngram_counts in ngram_counts_per_size.items():
            create_table_query = get_create_table_query(ngram_size)
            table_name = get_ngram_table_name(ngram_size)
            self.db_connection.execute(create_table_query)
            entries_to_insert = [ngram + (count, ) for ngram, count in ngram_counts.items()]
            entries_to_insert = ', '.join(map(str, entries_to_insert))
            insert_query = f"INSERT INTO {table_name} VALUES {entries_to_insert};"
            self.db_connection.execute(insert_query)

        # add the column
        compute_log_likelihood(self.db_connection, ngrams_of_size_count)

        for ngram_size, ngram_counts in ngram_counts_per_size.items():
            table_name = get_ngram_table_name(ngram_size)
            expected = [math.log(count) - math.log(ngrams_of_size_count[ngram_size]) for _, count in ngram_counts.items()]
            result = self.db_connection.sql(f'SELECT {fields.LOG_LIKELIHOOD} FROM {table_name}').fetchall()
            for e, r in zip(expected, result):
                r = r[0]
                self.assertAlmostEqual(e, r)


if __name__ == '__main__':
    unittest.main()
