import unittest

import duckdb
import pyarrow as pa

from src.aggregate_batch_ngram_counts import SQLField, get_create_table_query


class TestAggregateBatchNgramCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = duckdb.connect(':memory:')

    def test_get_create_table_query(self):
        # TODO
        new_table_name = 'new_table'
        key_fields = [
            SQLField(field_name='a', field_type='UINTEGER'),
            SQLField(field_name='b', field_type='UINTEGER')
        ]
        data_fields = [
            SQLField(field_name='x', field_type='DOUBLE'),
            SQLField(field_name='y', field_type='DOUBLE')
        ]
        create_query = get_create_table_query(new_table_name, key_fields, data_fields)
        self.connection.execute(create_query)
        insert_query = f'INSERT INTO {new_table_name} VALUES (1, 2, 0.1, 0.2);'
        self.connection.execute(insert_query)
        result = self.connection.sql(f'SELECT * FROM {new_table_name}').arrow()

        schema = pa.schema([
            pa.field('a', pa.uint32()),
            pa.field('b', pa.uint32()),
            pa.field('x', pa.float64()),
            pa.field('y', pa.float64())
        ])
        expected = pa.table({'a': [1], 'b': [2], 'x': [0.1], 'y': [0.2]}, schema=schema)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
