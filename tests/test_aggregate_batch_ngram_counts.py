import unittest

import duckdb
import pyarrow as pa

from src import fields
from src.aggregate_batch_ngram_counts import get_create_table_query, get_merge_and_add_counts_query
from src.utils import get_ngram_table_name, get_token_field_name


class MyTestCase(unittest.TestCase):
    schema = pa.schema([
            pa.field(get_token_field_name(0), pa.uint32()),
            pa.field(get_token_field_name(1), pa.uint32()),
            pa.field(fields.COUNT, pa.uint32()),
    ])
    ngram_size = 2

    def setUp(self) -> None:
        self.connection = duckdb.connect(':memory:')

    def tearDown(self) -> None:
        self.connection.close()

    def test_get_create_table_query(self):
        expected = pa.table({get_token_field_name(0): [1], get_token_field_name(1): [2],
                             fields.COUNT: [5]}, schema=self.schema)

        create_query = get_create_table_query(self.ngram_size)
        table_name = get_ngram_table_name(self.ngram_size)
        self.connection.execute(create_query)
        insert_query = f'INSERT INTO {table_name} VALUES (1, 2, 5);'
        self.connection.execute(insert_query)
        result = self.connection.sql(f'SELECT * FROM {table_name}').arrow()

        self.assertEqual(expected, result)

    def test_get_merge_and_add_counts_query(self):
        expected = {
            (1, 1, 5),
            (1, 2, 6),
            (2, 1, 7)
        }

        # initialize the table
        create_table_query = get_create_table_query(self.ngram_size)
        self.connection.execute(create_table_query)
        # add the first table
        table_to_insert1 = pa.table({
            get_token_field_name(0): [1, 2],
            get_token_field_name(1): [2, 1],
            fields.COUNT:   [6, 5]},
            schema=self.schema
        )
        query = get_merge_and_add_counts_query(ngram_size=self.ngram_size, table_to_insert='table_to_insert1')
        self.connection.execute(query)
        # add the second table (with and without overlap)
        table_to_insert2 = pa.table({
            get_token_field_name(0): [1, 2],
            get_token_field_name(1): [1, 1],
            fields.COUNT:   [5, 2]},
            schema=self.schema
        )
        query = get_merge_and_add_counts_query(ngram_size=self.ngram_size, table_to_insert='table_to_insert2')
        self.connection.execute(query)

        result = set(self.connection.sql(f'SELECT * FROM {get_ngram_table_name(self.ngram_size)}').fetchall())

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
