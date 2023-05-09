import unittest

import duckdb

from src import fields
from src.compute_pmi_score import compute_pmi_score_for_ngram_size
from src.naive_implementation import pmi_score
from src.utils import get_token_field_name, get_ngram_table_name
from tests.duckdb_test_case import DuckDBTestCase


# TODO: take the time with this to refactor the other tests, with utilities such as setting up a table and comparing
#   python objects and tables.
class MyTestCase(DuckDBTestCase):
    def test_compute_pmi_score(self):
        inputs = {
            get_token_field_name(0): [0, 1, 2],
            get_token_field_name(1): [0, 0, 0],
            fields.LOG_LIKELIHOOD: [0.1, 0.2, 0.3],
            fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM: [-0.1, -0.2, -0.3]
        }
        expected_pmi_scores = [
            pmi_score(0.1, -0.1),
            pmi_score(0.2, -0.2),
            pmi_score(0.3, -0.3)
        ]
        ngram_size = 2
        table_name = get_ngram_table_name(ngram_size)
        self.insert_pydict_as_table(inputs, table_name)

        compute_pmi_score_for_ngram_size(self.db_connection, ngram_size)

        actual_pmi_scores = self.fetch_column_as_list(fields.PMI_SCORE, table_name)
        self.assertListAlmostEqual(expected_pmi_scores, actual_pmi_scores)


if __name__ == '__main__':
    unittest.main()
