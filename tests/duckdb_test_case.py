import unittest

import duckdb
import pyarrow as pa


class DuckDBTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.db_connection = duckdb.connect(':memory:')

    def tearDown(self) -> None:
        self.db_connection.close()

    def insert_pydict_as_table(self, pydict: dict[str, list], table_name: str):
        pa_table = pa.table(pydict)
        query = f'CREATE TABLE {table_name} AS SELECT * FROM pa_table'
        self.db_connection.execute(query)

    def fetch_column_as_list(self, column_name: str, table_name: str) -> list:
        query = f'SELECT {column_name} FROM {table_name}'
        column_value_tuples = self.db_connection.sql(query).fetchall()
        column_values = [value_tuple[0] for value_tuple in column_value_tuples]
        return column_values

    def assertListAlmostEqual(self, expected: list[float], actual: list[float]):
        self.assertEqual(len(expected), len(actual))
        for v1, v2 in zip(expected, actual):
            self.assertAlmostEqual(v1, v2)

