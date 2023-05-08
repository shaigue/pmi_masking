import math
import shutil
import unittest
from pathlib import Path

import duckdb

from src.add_log_likelihood_column import add_log_likelihood_column
from src.aggregate_batch_ngram_counts import get_create_table_query
from src.utils import get_ngram_table_name


class TestAddLogLikelihoodColumn(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path('../test_temp')
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_something(self):
        database_file = self.temp_dir / 'ngram_data.duckdb'
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
        connection = duckdb.connect(str(database_file))
        for ngram_size, ngram_counts in ngram_counts_per_size.items():
            create_table_query = get_create_table_query(ngram_size)
            table_name = get_ngram_table_name(ngram_size)
            connection.execute(create_table_query)
            entries_to_insert = [ngram + (count, ) for ngram, count in ngram_counts.items()]
            entries_to_insert = ', '.join(map(str, entries_to_insert))
            insert_query = f"INSERT INTO {table_name} VALUES {entries_to_insert};"
            connection.execute(insert_query)
        connection.close()

        # add the column
        add_log_likelihood_column(database_file, ngrams_of_size_count)

        connection = duckdb.connect(str(database_file))
        for ngram_size, ngram_counts in ngram_counts_per_size.items():
            table_name = get_ngram_table_name(ngram_size)
            expected = [math.log(count) - math.log(ngrams_of_size_count[ngram_size]) for _, count in ngram_counts.items()]
            result = connection.sql(f'SELECT log_likelihood FROM {table_name}').fetchall()
            for e, r in zip(expected, result):
                r = r[0]
                self.assertAlmostEqual(e, r)
        connection.close()


if __name__ == '__main__':
    unittest.main()
