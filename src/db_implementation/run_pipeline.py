import shutil

import config
from src.experiment_config import ExperimentConfig
from src.db_implementation.compute_log_likelihood import compute_log_likelihood
from src.db_implementation.aggregate_ngram_counts import aggregate_ngram_counts
from src.db_implementation.compute_max_segmentation_log_likelihood_sum import compute_max_segmentation_log_likelihood_sum
from src.db_implementation.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.db_implementation.compute_pmi_score import compute_pmi_score
from src.db_implementation.count_ngrams_in_batches import count_ngrams_in_batches
from src.db_implementation.prune_low_count_ngrams import prune_low_count_ngrams
from src.load_dataset import load_and_tokenize_dataset
from src.utils import get_module_logger, get_file_size_bytes, \
    Ngram
from src.db_implementation.utils import read_total_ngrams_per_size, get_db_path, open_db_connection

logger = get_module_logger(__name__)


def get_experiment_name(experiment_config):
    """Extracts the name of the python configuration file"""
    return experiment_config.__name__.split('.')[-1]


def get_save_dir(experiment_name: str):
    return config.DATA_DIR / experiment_name


def run_pipeline(configuration: ExperimentConfig, clean_up: bool = True) -> list[Ngram]:
    """Runs the entire pipeline for creating the PMI masking vocabulary from a dataset, using a database."""
    # TODO: maybe save the pmi_masking_vocabulary to a file in the end?
    # TODO: add checkpoints? so we can resume if we were interrupted?
    # TODO: add some way to show progress. running this on a large dataset can take a lot of time.
    logger.info(f'start experiment: {configuration.name}')
    logger.info(f'n_workers={configuration.n_workers}')

    save_dir = get_save_dir(configuration.name)
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)
    db_connection = open_db_connection(save_dir)

    dataset = load_and_tokenize_dataset(dataset_name=configuration.dataset_name,
                                        tokenizer_name=configuration.tokenizer_name, n_workers=configuration.n_workers,
                                        n_samples=configuration.n_samples)
    count_ngrams_in_batches(
        tokenized_dataset=dataset,
        save_dir=save_dir,
        ngram_count_batch_size=configuration.ngram_count_batch_size,
        n_workers=configuration.n_workers,
        max_ngram_size=configuration.max_ngram_size,
        filter_ngram_count_threshold=configuration.min_count_batch_threshold,
    )
    aggregate_ngram_counts(save_dir, db_connection, configuration.max_ngram_size)

    db_connection.commit()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after aggregate counts: {db_size_bytes}')

    prune_low_count_ngrams(db_connection, configuration.max_ngram_size, configuration.min_count_threshold)
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, configuration.max_ngram_size)
    compute_pmi_score(db_connection, configuration.max_ngram_size)

    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, configuration.vocab_size,
                                                  configuration.ngram_size_to_vocab_percent)

    db_connection.close()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after pmi compute: {db_size_bytes}')

    if clean_up:
        shutil.rmtree(save_dir, ignore_errors=True)

    logger.info(f'end experiment: {configuration.name}')

    return pmi_masking_vocab


if __name__ == '__main__':
    from experiment_config import medium_size_bookcorpus
    from experiment_config import bookcorpus

    run_pipeline(medium_size_bookcorpus.config)
    run_pipeline(bookcorpus.config)
