import duckdb

from src import fields
from src.utils import validate_ngram_size_to_vocab_percent, compute_number_of_ngrams_per_size_in_vocab, Ngram, \
    get_token_fields_str, get_ngram_table_name


# TODO: complete and run the end-to-end test
# TODO: add logging, collect data for scalability analysis
# TODO: add to run_pipline
# TODO: run the entire thing
# TODO: do scalability analysis
# TODO: try to run on the entire bookcorpus dataset, and write methods to compare the resulting vocabulary and the
# TODO: prepare a test to run on bookcorpus and wikipedia, and check how similar is the resulting vocabulary to the one
#   published in by AI21 labs. send zach to run it in the server. make sure to capture logs, generated data,
#   and error messages, so I can examine those afterwards.
#   write detailed README.md before that with instructions on how to run.


def compute_pmi_masking_vocab_per_ngram_size(db_connection: duckdb.DuckDBPyConnection, ngram_size: int,
                                             ngrams_of_size_in_vocab: int) -> list[Ngram]:
    token_fields_str = get_token_fields_str(ngram_size)
    table_name = get_ngram_table_name(ngram_size)
    query = f'SELECT {token_fields_str} FROM {table_name} ' \
            f'ORDER BY {fields.PMI_SCORE} DESC ' \
            f'LIMIT {ngrams_of_size_in_vocab}'
    return db_connection.sql(query).fetchall()


def compute_pmi_masking_vocab(db_connection: duckdb.DuckDBPyConnection, vocab_size: int,
                              ngram_size_to_vocab_percent: dict[int, float]) -> list[Ngram]:
    """Computes the pmi masking vocabulary.
    :param db_connection: an open read/write connection to duckdb database.
    :param vocab_size: the size of the resulting vocabulary.
    :param ngram_size_to_vocab_percent: dictionary mapping ngrams size to the percentage of ngrams of that size in the
        resulting vocabulary.
        for example, ngram_size_to_vocab_percent={2: 30, 3: 30, 4:40} means that the resulting vocabulary will be 30%
        ngrams of size 2, 30% ngrams of size 3 and 40% ngrams of size 4.
    :return: list containing the ngrams selected to go into the masking vocabulary.
    """
    validate_ngram_size_to_vocab_percent(ngram_size_to_vocab_percent)
    number_of_ngrams_per_size_in_vocab = compute_number_of_ngrams_per_size_in_vocab(ngram_size_to_vocab_percent,
                                                                                    vocab_size)
    vocab = []
    for ngram_size, ngrams_of_size_in_vocab in number_of_ngrams_per_size_in_vocab.items():
        vocab += compute_pmi_masking_vocab_per_ngram_size(
            db_connection,
            ngram_size,
            ngrams_of_size_in_vocab,
        )
    return vocab
