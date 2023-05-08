import unittest

import duckdb
import pyarrow as pa

from src.aggregate_batch_ngram_counts import get_create_table_query, get_merge_and_add_counts_query
from src.utils import get_ngram_table_name


class TestAggregateBatchNgramCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = duckdb.connect(':memory:')

    def tearDown(self) -> None:
        self.connection.close()

    def test_get_create_table_query(self):
        ngram_size = 2
        schema = pa.schema([
            pa.field('token_0', pa.uint32()),
            pa.field('token_1', pa.uint32()),
            pa.field('count', pa.uint32()),
        ])
        expected = pa.table({'token_0': [1], 'token_1': [2], 'count': [5]}, schema=schema)

        create_query = get_create_table_query(ngram_size)
        table_name = get_ngram_table_name(ngram_size)
        self.connection.execute(create_query)
        insert_query = f'INSERT INTO {table_name} VALUES (1, 2, 5);'
        self.connection.execute(insert_query)
        result = self.connection.sql(f'SELECT * FROM {table_name}').arrow()

        self.assertEqual(expected, result)

    def test_get_merge_and_add_counts_query(self):
        ngram_size = 2
        schema = pa.schema([
            pa.field('token_0', pa.uint32()),
            pa.field('token_1', pa.uint32()),
            pa.field('count', pa.uint32()),
        ])
        expected = {
            (1, 1, 5),
            (1, 2, 6),
            (2, 1, 7)
        }

        # initialize the table
        create_table_query = get_create_table_query(ngram_size)
        self.connection.execute(create_table_query)
        # add the first table
        table_to_insert1 = pa.table({
            'token_0': [1, 2],
            'token_1': [2, 1],
            'count':   [6, 5]},
            schema=schema
        )
        query = get_merge_and_add_counts_query(ngram_size=ngram_size, table_to_insert='table_to_insert1')
        self.connection.execute(query)
        # add the second table (with and without overlap)
        table_to_insert2 = pa.table({
            'token_0': [1, 2],
            'token_1': [1, 1],
            'count':   [5, 2]},
            schema=schema
        )
        query = get_merge_and_add_counts_query(ngram_size=ngram_size, table_to_insert='table_to_insert2')
        self.connection.execute(query)

        result = set(self.connection.sql(f'SELECT * FROM {get_ngram_table_name(ngram_size)}').fetchall())

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
