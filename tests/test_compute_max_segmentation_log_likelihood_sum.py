import unittest

import duckdb
import pyarrow as pa

from src.compute_max_segmentation_log_likelihood_sum import get_join_condition, \
    compute_max_segmentation_log_likelihood_sum
from src.utils import get_ngram_table_name, Field


class TestComputeMaxSegmentationLogLikelihoodSum(unittest.TestCase):
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
        expected1 = '(t.token_0 = t1.token_0 AND t.token_1 = t1.token_1)'
        expected2 = '(t.token_2 = t2.token_0 AND t.token_3 = t2.token_1 AND t.token_4 = t2.token_2)'

        result1 = get_join_condition(table_name, table1_name, sub_ngram_size1, 0)
        result2 = get_join_condition(table_name, table2_name, sub_ngram_size2, sub_ngram_size1)
        self.assertEqual(expected1, result1)
        self.assertEqual(expected2, result2)

    # TODO: maybe do a 'test_utils' module, especially for converting table data into python objects
    def test_compute_max_segmentation_log_likelihood_sum(self):
        max_ngram_size = 3
        table_of_size_1 = {
            'token_0':        [0, 1],
            'log_likelihood': [-0.1, -0.2]
        }
        table_of_size_2 = {
            'token_0':        [0, 0, 1],
            'token_1':        [0, 1, 0],
            'log_likelihood': [-0.2, -0.3, -0.15]
        }
        expected_max_segmentation_log_likelihood_sum_of_size_2 = [
            -0.1 + -0.1,
            -0.1 + -0.2,
            -0.2 + -0.1
        ]
        table_of_size_3 = {
            'token_0':        [0, 0],
            'token_1':        [0, 1],
            'token_2':        [0, 0],
            'log_likelihood': [-0.3, -0.4]
        }
        expected_max_segmentation_log_likelihood_sum_of_size_3 = [
            max(-0.1 + -0.1 + -0.1, -0.2 + -0.1, -0.1 + -0.2),
            max(-0.1 + -0.2 + -0.1, -0.3 + -0.1, -0.1 + -0.15)
        ]
        # prepare the input such that it can be used by the function

        # TODO: this is a utility function that might be used in other places
        def create_table(db_connection: duckdb.DuckDBPyConnection, table_name: str, data: dict[str, list]):
            pa_table = pa.table(data)
            query = f'CREATE TABLE {table_name} AS SELECT * FROM pa_table'
            db_connection.execute(query)

        create_table(self.db_connection, get_ngram_table_name(1), table_of_size_1)
        create_table(self.db_connection, get_ngram_table_name(2), table_of_size_2)
        create_table(self.db_connection, get_ngram_table_name(3), table_of_size_3)

        compute_max_segmentation_log_likelihood_sum(self.db_connection, max_ngram_size)

        # TODO: consider extracting this as a utility function
        def load_data_from_db(db_connection: duckdb.DuckDBPyConnection, table_name: str, field: Field):
            query = f'SELECT {field} FROM {table_name}'
            return [v[0] for v in db_connection.sql(query).fetchall()]

        # then load the data from the memory table and compare the values
        # compare by size 2
        result_max_segmentation_log_likelihood_sum_of_size_2 = load_data_from_db(
            self.db_connection,
            get_ngram_table_name(2),
            Field.max_segmentation_log_likelihood_sum
        )
        for v1, v2 in zip(expected_max_segmentation_log_likelihood_sum_of_size_2,
                          result_max_segmentation_log_likelihood_sum_of_size_2):
            self.assertAlmostEqual(v1, v2)
        result_max_segmentation_log_likelihood_sum_of_size_3 = load_data_from_db(
            self.db_connection,
            get_ngram_table_name(3),
            Field.max_segmentation_log_likelihood_sum
        )
        for v1, v2 in zip(expected_max_segmentation_log_likelihood_sum_of_size_3,
                          result_max_segmentation_log_likelihood_sum_of_size_3):
            self.assertAlmostEqual(v1, v2)


if __name__ == '__main__':
    unittest.main()
