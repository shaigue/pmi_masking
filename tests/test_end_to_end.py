"""end-to-end test comparing the naive implementation and the DuckDB based implementation"""
import shutil
import unittest
from typing import Union

import duckdb

from src import fields
from src.naive_implementation import run_pipeline_naive_with_parameters
from src.run_pipeline import run_pipeline_with_experiment_config
from src.utils import read_total_ngrams_per_size, open_db_connection, Ngram, get_ngram_table_name, get_token_fields_str
import experiment_config.end_to_end_test as parameters


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        shutil.rmtree(parameters.save_dir, ignore_errors=True)

    def assertDictAlmostEqual(self, d1: dict, d2: dict):
        self.assertEqual(set(d1.keys()), set(d2.keys()))
        for k in d1.keys():
            self.assertAlmostEqual(d1[k], d2[k])

    @staticmethod
    def read_column_as_dict(db_connection: duckdb.DuckDBPyConnection, column: str,
                            max_ngram_size: int, skip_unigrams: bool) -> dict[Ngram, Union[int, float]]:
        column_as_dict = {}
        start_ngram_size = 2 if skip_unigrams else 1
        for ngram_size in range(start_ngram_size, max_ngram_size + 1):
            table_name = get_ngram_table_name(ngram_size)
            token_fields_str = get_token_fields_str(ngram_size)
            select_query = f'SELECT {token_fields_str}, {column} FROM {table_name}'
            for result_tuple in db_connection.sql(select_query).fetchall():
                ngram = result_tuple[:-1]
                value = result_tuple[-1]
                column_as_dict[ngram] = value
        return column_as_dict

    def assertColumnEqual(self, naive_result: dict, db_connection: duckdb.DuckDBPyConnection, column: str) -> None:
        ngram_lens = [len(ngram) for ngram in naive_result[column].keys()]
        max_ngram_size = max(ngram_lens)
        skip_unigrams = 1 not in ngram_lens
        db_result = self.read_column_as_dict(db_connection, column, max_ngram_size, skip_unigrams)
        self.assertDictAlmostEqual(naive_result[column], db_result)

    def test_end_to_end(self):
        pmi_masking_vocab_db = run_pipeline_with_experiment_config(parameters)
        naive_result = run_pipeline_naive_with_parameters(parameters)

        total_ngrams_per_size_db = read_total_ngrams_per_size(parameters.save_dir)
        self.assertDictEqual(naive_result['total_ngrams_per_size'], total_ngrams_per_size_db)

        db_connection = open_db_connection(parameters.save_dir)
        self.assertColumnEqual(naive_result, db_connection, fields.COUNT)
        self.assertColumnEqual(naive_result, db_connection, fields.LOG_LIKELIHOOD)
        self.assertColumnEqual(naive_result, db_connection, fields.MAX_SEGMENTATION_LOG_LIKELIHOOD_SUM)
        self.assertColumnEqual(naive_result, db_connection, fields.PMI_SCORE)
        db_connection.close()

        self.assertEqual(set(naive_result['pmi_masking_vocab']), set(pmi_masking_vocab_db))


if __name__ == '__main__':
    unittest.main()
