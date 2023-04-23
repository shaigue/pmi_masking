import unittest
from collections import Counter
from io import StringIO
from math import log
from unittest import skip

from pmi_masking.pmi import \
    iter_ngram_segmentations, \
    count_ngrams_of_size_in_sequence_update, save_ngram_counter_to_file, load_ngram_counter_from_file, \
    save_ngrams_of_size_counter_to_file, load_ngrams_of_size_counter_from_file
from pmi_masking.count_ngrams_in_chunks import count_ngrams_in_sequence_update


class TestPMI(unittest.TestCase):
    def test_count_ngrams_in_sequence_update(self):
        tokens = [1, 1, 1, 0, 1, 0, 2, 0, 2, 1]
        max_ngram_len = 3
        expected = {
            (1, ): 5,
            (0, ): 3,
            (2, ): 2,
            (1, 1): 2,
            (1, 1, 1): 1,
            (1, 0): 2,
            (1, 1, 0): 1,
            (0, 1): 1,
            (1, 0, 1): 1,
            (0, 1, 0): 1,
            (0, 2): 2,
            (1, 0, 2): 1,
            (2, 0): 1,
            (0, 2, 0): 1,
            (2, 0, 2): 1,
            (2, 1): 1,
            (0, 2, 1): 1
        }
        result = Counter()
        count_ngrams_in_sequence_update(result, tokens, max_ngram_len, ngram_to_str_func)
        self.assertEqual(expected, result)

    def test_count_ngrams_of_size_in_sequence_update(self):
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_ngram_size = 5
        expected = {
            1: 10,
            2: 9,
            3: 8,
            4: 7,
            5: 6

        }
        result = Counter()
        count_ngrams_of_size_in_sequence_update(result, sequence, max_ngram_size)
        self.assertEqual(expected, result)

    def test_iter_ngram_segmentations(self):
        ngram = (1, 2, 3, 4)
        expected_segmentations = [
            [(1, ), (2, ), (3, ), (4, )],
            [(1, 2), (3, ), (4, )],
            [(1, ), (2, 3), (4, )],
            [(1, ), (2, ), (3, 4)],
            [(1, 2), (3, 4)],
            [(1, 2, 3), (4, )],
            [(1, ), (2, 3, 4)],
        ]
        for segmentation in iter_ngram_segmentations(ngram):
            self.assertIn(segmentation, expected_segmentations)
            expected_segmentations.remove(segmentation)

        self.assertEqual(len(expected_segmentations), 0)

    def test_save_load_ngram_counter_to_file(self):
        counter = Counter({
            (1, 2, 3): 5,
            (1, ): 7,
            (3, 2, 1): 2,
        })
        file = StringIO()
        save_ngram_counter_to_file(counter, file)
        file.seek(0)
        result = load_ngram_counter_from_file(file)
        self.assertEqual(counter, result)

    def test_save_load_ngrams_of_size_counter_to_file(self):
        counter = Counter({
            1: 5,
            2: 4,
            3: 3,
            4: 2,
            5: 1
        })
        file = StringIO()
        save_ngrams_of_size_counter_to_file(counter, file)
        file.seek(0)
        result = load_ngrams_of_size_counter_from_file(file)
        self.assertEqual(counter, result)

    @skip('compute_corpus_pmi not yet stable.')
    def test_compute_corpus_pmi(self):
        tokens = [1, 1, 1, 0, 1, 0, 2, 0, 2, 1]
        max_ngram_len = 3
        corpus_pmi_dict = compute_corpus_pmi(tokens, max_ngram_len)
        expected_pmi_values = {
            (1, 1, 1): min(log((1 / 8) / ((5 / 10) * (5 / 10) * (5 / 10))),
                           log((1 / 8) / ((2 / 9) * (5 / 10)))),
            (0, 1): log((1 / 9) / ((3 / 10) * (5 / 10))),
            (1, 1): log((2 / 9) / ((5 / 10) * (5 / 10))),
        }
        for ngram, pmi_score in expected_pmi_values.items():
            self.assertAlmostEqual(pmi_score, corpus_pmi_dict[ngram])


if __name__ == '__main__':
    unittest.main()
