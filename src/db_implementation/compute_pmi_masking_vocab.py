import duckdb

from src.db_implementation import fields
from src.utils import validate_ngram_size_to_vocab_percent, compute_number_of_ngrams_per_size_in_vocab, Ngram, \
    get_module_logger
from src.db_implementation.utils import get_ngram_table_name, get_token_fields_str

logger = get_module_logger(__name__)


def compute_pmi_masking_vocab_per_ngram_size(db_connection: duckdb.DuckDBPyConnection, ngram_size: int,
                                             ngrams_of_size_in_vocab: int) -> list[Ngram]:
    """Takes the ngrams with the highest pmi score of a given ngrams size.

    :param db_connection: open connection to the DB
    :param ngram_size: size of the ngrams that are interested in
    :param ngrams_of_size_in_vocab: number of ngrams of that size that we put in our vocabulary
    :return: List of the ngrams of that size with the highest PMI scores
    """
    token_fields_str = get_token_fields_str(ngram_size)
    table_name = get_ngram_table_name(ngram_size)
    query = f'SELECT {token_fields_str} FROM {table_name} ' \
            f'ORDER BY {fields.PMI_SCORE} DESC ' \
            f'LIMIT {ngrams_of_size_in_vocab}'
    return db_connection.sql(query).fetchall()


def compute_pmi_masking_vocab(db_connection: duckdb.DuckDBPyConnection, vocab_size: int,
                              ngram_size_to_vocab_percent: dict[int, float]) -> list[Ngram]:
    """Computes the PMI masking vocabulary.

    :param db_connection: an open read/write connection to duckdb database.
    :param vocab_size: total size of the resulting vocabulary.
    :param ngram_size_to_vocab_percent: dictionary mapping ngrams size to the percentage of ngrams of that size in the
        resulting vocabulary.
        for example, ngram_size_to_vocab_percent={2: 30, 3: 30, 4:40} means that the resulting vocabulary will be 30%
        ngrams of size 2, 30% ngrams of size 3 and 40% ngrams of size 4.
    :return: list containing the ngrams selected to go into the masking vocabulary.
    """
    logger.info('start')
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
    logger.info('end')
    return vocab
