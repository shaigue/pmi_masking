import shutil
from pathlib import Path

import config
from src.db_implementation.compute_log_likelihood import compute_log_likelihood
from src.db_implementation.aggregate_ngram_counts import aggregate_ngram_counts
from src.db_implementation.compute_max_segmentation_log_likelihood_sum import compute_max_segmentation_log_likelihood_sum
from src.db_implementation.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.db_implementation.compute_pmi_score import compute_pmi_score
from src.db_implementation.count_ngrams_in_batches import count_ngrams_in_batches
from src.db_implementation.filter_low_count_ngrams import filter_low_count_ngrams
from src.load_dataset import load_bookcorpus_dataset
from src.utils import read_total_ngrams_per_size, open_db_connection, get_module_logger, get_file_size_bytes, \
    get_db_path, Ngram

logger = get_module_logger(__name__)


def run_pipeline(max_ngram_size: int, min_count_threshold: int, vocab_size: int,
                 ngram_size_to_vocab_percent: dict[int, float], ngram_count_batch_size: int,
                 filter_ngram_count_threshold: int, save_dir: Path, n_workers: int = None, n_samples: int = None,
                 **kwargs) -> list[Ngram]:
    # TODO: make this work with various datasets. currently only works for bookcorpus.
    # TODO: add checkpoints? so we can resume if we were interrupted?
    # TODO: add some way to show progress. running this on a large dataset can take a lot of time.
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)
    db_connection = open_db_connection(save_dir)

    dataset = load_bookcorpus_dataset(n_samples=n_samples, n_workers=n_workers)
    count_ngrams_in_batches(
        tokenized_dataset=dataset,
        save_dir=save_dir,
        ngram_count_batch_size=ngram_count_batch_size,
        n_workers=n_workers,
        max_ngram_size=max_ngram_size,
        filter_ngram_count_threshold=filter_ngram_count_threshold,
    )
    aggregate_ngram_counts(save_dir, max_ngram_size, db_connection)

    db_connection.commit()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after aggregate counts: {db_size_bytes}')

    filter_low_count_ngrams(max_ngram_size, db_connection, min_count_threshold)
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, max_ngram_size)
    compute_pmi_score(db_connection, max_ngram_size)

    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, vocab_size, ngram_size_to_vocab_percent)

    db_connection.close()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after pmi compute: {db_size_bytes}')

    return pmi_masking_vocab


def get_experiment_name(experiment_config):
    return experiment_config.__name__.split('.')[-1]


def get_save_dir(experiment_name: str):
    return config.DATA_DIR / experiment_name


def run_pipeline_with_experiment_config(experiment_config, clean_up: bool = True):
    # TODO: make sure to update the run_pipeline_naive version
    # TODO: maybe save the pmi_masking_vocabulary to a file in the end?
    experiment_name = get_experiment_name(experiment_config)
    save_dir = get_save_dir(experiment_name)
    logger.info(f'start experiment_config: {experiment_name}')
    pmi_masking_vocab = run_pipeline(save_dir=save_dir, **experiment_config.__dict__)
    logger.info(f'end experiment_config: {experiment_name}')
    if clean_up:
        shutil.rmtree(save_dir, ignore_errors=True)
    return pmi_masking_vocab


if __name__ == '__main__':
    # from experiment_config import medium_size_bookcorpus
    # run_pipeline_with_experiment_config(medium_size_bookcorpus)
    from experiment_config import bookcorpus
    run_pipeline_with_experiment_config(bookcorpus)
