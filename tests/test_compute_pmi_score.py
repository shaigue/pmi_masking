import unittest

from src.db_implementation import fields
from src.db_implementation.compute_pmi_score import compute_pmi_score_for_ngram_size
from src.naive_implementation import pmi_score
from src.db_implementation.utils import get_ngram_table_name
from src.db_implementation.fields import get_token_field_name
from tests.duckdb_test_case import DuckDBTestCase


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
