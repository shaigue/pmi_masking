import unittest

import duckdb
import pyarrow as pa

from src.db_implementation import fields
from src.db_implementation.compute_max_segmentation_log_likelihood_sum import get_join_condition, \
    compute_max_segmentation_log_likelihood_sum
from src.db_implementation.utils import get_ngram_table_name
from src.db_implementation.fields import get_token_field_name


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.db_connection = duckdb.connect(':memory:')

    def tearDown(self) -> None:
        self.db_connection.close()

    def test_get_join_condition(self):
        ngram_size = 5
        table_name = 't'
        table1_name = 't1'
        table2_name = 't2'
        sub_ngram_size1 = 2
        sub_ngram_size2 = ngram_size - sub_ngram_size1
        expected1 = f'(t.{get_token_field_name(0)} = t1.{get_token_field_name(0)} ' \
                    f'AND t.{get_token_field_name(1)} = t1.{get_token_field_name(1)})'
        expected2 = f'(t.{get_token_field_name(2)} = t2.{get_token_field_name(0)} ' \
                    f'AND t.{get_token_field_name(3)} = t2.{get_token_field_name(1)} ' \
                    f'AND t.{get_token_field_name(4)} = t2.{get_token_field_name(2)})'

        result1 = get_join_condition(table_name, table1_name, sub_ngram_size1, 0)
        result2 = get_join_condition(table_name, table2_name, sub_ngram_size2, sub_ngram_size1)
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    def test_compute_max_segmentation_log_likelihood_sum(self):
        max_ngram_size = 3
        table_of_size_1 = {
            get_token_field_name(0): [0, 1],
            fields.LOG_LIKELIHOOD: [-0.1, -0.2]
        }
        table_of_size_2 = {
            get_token_field_name(0): [0, 0, 1],
            get_token_field_name(1): [0, 1, 0],
            fields.LOG_LIKELIHOOD: [-0.2, -0.3, -0.15]
        }
        expected_max_segmentation_log_likelihood_sum_of_size_2 = [
            -0.1 + -0.1,
            -0.1 + -0.2,
            -0.2 + -0.1
        ]
        table_of_size_3 = {
            get_token_field_name(0):        [0, 0],
            get_token_field_name(1):        [0, 1],
            get_token_field_name(2):        [0, 0],
            fields.LOG_LIKELIHOOD: [-0.3, -0.4]
        }
        expected_max_segmentation_log_likelihood_sum_of_size_3 = [
            max(-0.1 + -0.1 + -0.1, -0.2 + -0.1, -0.1 + -0.2),
            max(-0.1 + -0.2 + -0.1, -0.3 + -0.1, -0.1 + -0.15)
        ]

        def create_table(db_connection: duckdb.DuckDBPyConnection, table_name: str, data: dict[str, list]):
            pa_table = pa.table(data)
            query = f'CREATE TABLE {table_name} AS SELECT * FROM pa_table'
            db_connection.execute(query)

        create_table(self.db_connection, get_ngram_table_name(1), table_of_size_1)
        create_table(self.db_connection, get_ngram_table_name(2), table_of_size_2)
        create_table(self.db_connection, get_ngram_table_name(3), table_of_size_3)

        compute_max_segmentation_log_likelihood_sum(self.db_connection, max_ngram_size)

        # TODO: consider extracting this as a utility function
        def load_data_from_db(db_connection: duckdb.DuckDBPyConnection, table_name: str, field: str):
            query = f'SELECT {field} FROM {table_name}'
            return [v[0] for v in db_connection.sql(query).fetchall()]

        # then load the data from the memory table and compare the values
        # compare by size 2
        result_max_segmentation_log_likelihood_sum_of_size_2 = load_data_from_db(
            self.db_connection,
            get_ngram_table_name(2),
            fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM
        )
        for v1, v2 in zip(expected_max_segmentation_log_likelihood_sum_of_size_2,
                          result_max_segmentation_log_likelihood_sum_of_size_2):
            self.assertAlmostEqual(v1, v2)
        result_max_segmentation_log_likelihood_sum_of_size_3 = load_data_from_db(
            self.db_connection,
            get_ngram_table_name(3),
            fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM
        )
        for v1, v2 in zip(expected_max_segmentation_log_likelihood_sum_of_size_3,
                          result_max_segmentation_log_likelihood_sum_of_size_3):
            self.assertAlmostEqual(v1, v2)


if __name__ == '__main__':
    unittest.main()
