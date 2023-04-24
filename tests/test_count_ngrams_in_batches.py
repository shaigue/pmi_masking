import unittest
from collections import Counter

import pyarrow as pa

from src.count_ngrams_in_batches import count_ngrams_in_batch, convert_ngram_counter_to_pa_table, \
    count_total_ngrams_of_size_


class TestCountNgramsInBatches(unittest.TestCase):
    def test_count_ngrams_in_batch(self):
        batch = [
            [1, 1, 1, 0, 1, 0, 2, 0, 2, 1]
        ]
        max_ngram_len = 3
        expected = {
            1: {
                (1,): 5,
                (0,): 3,
                (2,): 2,
            },
            2: {
                (0, 1): 1,
                (1, 0): 2,
                (1, 1): 2,
                (0, 2): 2,
                (2, 0): 1,
                (2, 1): 1,
            },
            3: {
                (1, 1, 1): 1,
                (1, 1, 0): 1,
                (1, 0, 1): 1,
                (0, 1, 0): 1,
                (1, 0, 2): 1,
                (0, 2, 0): 1,
                (2, 0, 2): 1,
                (0, 2, 1): 1
            },
        }
        result = count_ngrams_in_batch(batch, max_ngram_len)
        self.assertEqual(expected, result)

    def test_convert_ngram_counter_to_pa_table(self):
        counter = Counter({
            (1, 1, 1): 1,
            (1, 1, 0): 2,
            (1, 0, 1): 1,
            (0, 1, 0): 3,
            (1, 0, 2): 1,
            (0, 2, 0): 4,
        })
        ngram_size = 3
        filter_ngram_count_threshold = 1
        result = convert_ngram_counter_to_pa_table(counter, ngram_size, filter_ngram_count_threshold)
        expected = pa.table(
            data={
                'token_1': [1, 1, 1, 0, 1, 0],
                'token_2': [1, 1, 0, 1, 0, 2],
                'token_3': [1, 0, 1, 0, 2, 0],
                'count':   [1, 2, 1, 3, 1, 4]
            }
        )
        self.assertEqual(expected, result)

    def test_count_total_ngrams_of_size_(self):
        input_ids = pa.array([
            [1, 2, 3, 4],
            [1],
            [1, 2, 3, 4, 5, 6]
        ])
        max_ngram_size = 3
        expected = {
            1: 4 + 1 + 6,
            2: 3 + 0 + 5,
            3: 2 + 0 + 4,
        }
        result = count_total_ngrams_of_size_(input_ids, max_ngram_size)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
