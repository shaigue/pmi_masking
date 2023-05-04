import unittest
from math import log

from src.naive_implementation import count_total_ngrams_per_size, count_ngrams, iter_ngram_segmentations, \
    compute_pmi_scores_from_tokenized_samples


class TestNaiveImplementation(unittest.TestCase):
    def test_count_total_ngrams_per_size(self):
        tokenized_samples = [
            [1, 1, 1, 0, 1],
            [2, 2, 2]
        ]
        max_ngram_size = 5
        expected = {
            1: 5 + 3,
            2: 4 + 2,
            3: 3 + 1,
            4: 2 + 0,
            5: 1 + 0,
        }
        result = count_total_ngrams_per_size(tokenized_samples, max_ngram_size)
        self.assertEqual(expected, result)

    def test_count_ngrams_1(self):
        tokenized_samples = [
            [1, 1, 1, 0, 1, 0, 2, 0, 2, 1]
        ]
        max_ngram_size = 3
        expected = {
            (0, ): 3,
            (1, ): 5,
            (2, ): 2,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 2,
            (1, 1): 2,
            (2, 0): 1,
            (2, 1): 1,
            (0, 1, 0): 1,
            (0, 2, 0): 1,
            (0, 2, 1): 1,
            (1, 0, 1): 1,
            (1, 0, 2): 1,
            (1, 1, 0): 1,
            (1, 1, 1): 1,
            (2, 0, 2): 1,
        }
        result = count_ngrams(tokenized_samples, max_ngram_size)
        self.assertEqual(expected, result)

    def test_count_ngrams_2(self):
        tokenized_samples = [
            [1, 1, 1, 0, 1],
            [2, 2, 2]
        ]
        max_ngram_size = 5
        expected = {
            (0, ): 1,
            (1, ): 4,
            (2, ): 3,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 2,
            (2, 2): 2,
            (1, 0, 1): 1,
            (1, 1, 0): 1,
            (1, 1, 1): 1,
            (2, 2, 2): 1,
            (1, 1, 0, 1): 1,
            (1, 1, 1, 0): 1,
            (1, 1, 1, 0, 1): 1
        }
        result = count_ngrams(tokenized_samples, max_ngram_size)
        self.assertEqual(expected, result)

    def test_iter_ngram_segmentations(self):
        ngram = (1, 2, 3, 4)
        expected = [
            [(1, ), (2, ), (3, ), (4, )],
            [(1, 2), (3, ), (4, )],
            [(1, ), (2, 3), (4, )],
            [(1, ), (2, ), (3, 4)],
            [(1, 2), (3, 4)],
            [(1, 2, 3), (4, )],
            [(1, ), (2, 3, 4)],
        ]
        result = list(iter_ngram_segmentations(ngram))
        for x in result:
            self.assertIn(x, expected, f'{x} appears in result but not in expected.')
        for x in expected:
            self.assertIn(x, result, f'{x} appears in expected but not in result')

    def test_compute_pmi_scores_from_tokenized_samples(self):
        tokenized_samples = [[1, 1, 1, 0, 1, 0, 2, 0, 2, 1]]
        max_ngram_size = 3
        expected = {
            (1, 1, 1): min(
                log(((1 / 8) ** 2) / ((5 / 10) * (5 / 10) * (5 / 10))),
                log(((1 / 8) ** 2) / ((2 / 9) * (5 / 10))),
            ),
            (0, 1): log(((1 / 9) ** 2) / ((3 / 10) * (5 / 10))),
            (1, 1): log(((2 / 9) ** 2) / ((5 / 10) * (5 / 10))),
        }
        result = compute_pmi_scores_from_tokenized_samples(tokenized_samples, max_ngram_size)
        for ngram, pmi_score in expected.items():
            self.assertAlmostEqual(pmi_score, result[ngram])


if __name__ == '__main__':
    unittest.main()
