import platform
import shutil
from typing import Optional

import psutil

from src.db_implementation.compute_log_likelihood import compute_log_likelihood
from src.db_implementation.aggregate_ngram_counts import aggregate_ngram_counts
from src.db_implementation.compute_max_segmentation_log_likelihood_sum import \
    compute_max_segmentation_log_likelihood_sum
from src.db_implementation.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.db_implementation.compute_pmi_score import compute_pmi_score
from src.db_implementation.count_ngrams_in_batches import count_ngrams_in_batches
from src.db_implementation.prune_low_count_ngrams import prune_low_count_ngrams
from src.load_dataset import load_and_tokenize_dataset
from src.utils import get_module_logger, get_file_size_bytes, Ngram
from src.db_implementation.utils import read_total_ngrams_per_size, get_db_path, open_db_connection, get_save_dir, \
    get_vocab_file

logger = get_module_logger(__name__)


def log_system_info():
    """Logs system information for creating performance report"""
    logger.info(f'os: {platform.platform()}')
    logger.info(f'processor: {platform.processor()}')
    logger.info(f'RAM_size: {psutil.virtual_memory().total}')


def run_pipeline(experiment_name: str, dataset_name: str, tokenizer_name: str,
                 max_ngram_size: int, min_count_threshold: int, vocab_size: int,
                 ngram_size_to_vocab_percent: dict[int, float], ngram_count_batch_size: int,
                 min_count_batch_threshold: int, n_workers: int, tokenizer_batch_size: int, n_samples: Optional[int],
                 clean_up: bool = True, save_vocab_to_file: bool = True) -> list[Ngram]:
    """Runs the entire pipeline for creating the PMI masking vocabulary from a dataset, using a database.

    :param experiment_name: name of the experiment
    :param dataset_name: name of the dataset to use
    :param tokenizer_name: name of the tokenizer to use
    :param max_ngram_size: maximal size of ngram to consider
    :param min_count_threshold: prune ngrams that occur less than that in the entire dataset
    :param vocab_size: number of ngrams in the resulting vocabulary
    :param ngram_size_to_vocab_percent: percent of ngrams per size in the resulting vocabulary
    :param ngram_count_batch_size: batch size in the distributed ngram counting phase
    :param min_count_batch_threshold: prune ngrams that occur less than that in the entire dataset
    :param n_workers: number of processes to use
    :param tokenizer_batch_size: size of the tokenizer batches
    :param n_samples: if not None, only the first `n_samples` of the dataset will be used.
    :param clean_up: if True, the database will be deleted after the pmi masking vocabulary is generated.
        otherwise, it will not be deleted.
    :param save_vocab_to_file: if True, the resulting PMI masking vocabulary will be saved to file, after converting
        back the token-ids to strings. Otherwise, it will not be saved.
    :return: a list of the generated pmi masking vocabulary (ngrams are represented with token ids).
    """
    logger.info(f'start experiment: {experiment_name}')
    logger.info(f'n_workers: {n_workers}')
    logger.info(f'dataset: {dataset_name}')
    log_system_info()

    save_dir = get_save_dir(experiment_name)
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True, parents=True)
    db_connection = open_db_connection(save_dir)

    dataset, tokenizer = load_and_tokenize_dataset(dataset_name=dataset_name, tokenizer_name=tokenizer_name,
                                                   tokenizer_batch_size=tokenizer_batch_size, n_samples=n_samples)
    count_ngrams_in_batches(
        tokenized_dataset=dataset,
        save_dir=save_dir,
        ngram_count_batch_size=ngram_count_batch_size,
        n_workers=n_workers,
        max_ngram_size=max_ngram_size,
        filter_ngram_count_threshold=min_count_batch_threshold,
    )
    aggregate_ngram_counts(save_dir, db_connection, max_ngram_size)

    db_connection.commit()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after aggregate counts: {db_size_bytes}')

    prune_low_count_ngrams(db_connection, max_ngram_size, min_count_threshold)
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, max_ngram_size)
    compute_pmi_score(db_connection, max_ngram_size)

    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, vocab_size, ngram_size_to_vocab_percent)

    db_connection.close()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after pmi compute: {db_size_bytes}')

    if clean_up:
        shutil.rmtree(save_dir, ignore_errors=True)

    if save_vocab_to_file:
        lines = tokenizer.batch_decode(pmi_masking_vocab)
        vocab_file = get_vocab_file(experiment_name)
        with vocab_file.open('w') as f:
            for line in lines:
                line += '\n'
                f.write(line)

    logger.info(f'end experiment: {experiment_name}')

    return pmi_masking_vocab
