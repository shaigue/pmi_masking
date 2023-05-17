import unittest

from src.db_implementation.compute_pmi_masking_vocab import compute_pmi_masking_vocab_per_ngram_size
from src.db_implementation.utils import get_ngram_table_name
from tests.duckdb_test_case import DuckDBTestCase


class MyTestCase(DuckDBTestCase):
    def test_compute_pmi_masking_vocab_per_ngram_size(self):
        ngram_size = 2
        ngrams_of_size_in_vocab = 5
        data = {
            'token_0':     [0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            'token_1':     [0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            'pmi_score':   [.0, .1, .02, .3, .04, .5, .06, .7, .08, .9]
        }
        expected = [(9, 9), (7, 7), (5, 5), (3, 3), (1, 1)]

        table_name = get_ngram_table_name(ngram_size)
        self.insert_pydict_as_table(data, table_name)
        actual = compute_pmi_masking_vocab_per_ngram_size(self.db_connection, ngram_size, ngrams_of_size_in_vocab)
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
