import shutil
from pathlib import Path

from src.compute_log_likelihood import compute_log_likelihood
from src.aggregate_batch_ngram_counts import aggregate_batch_ngram_counts
from src.compute_max_segmentation_log_likelihood_sum import compute_max_segmentation_log_likelihood_sum
from src.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.compute_pmi_score import compute_pmi_score
from src.count_ngrams_in_batches import count_ngrams_in_batches
from src.delete_low_count_ngrams import delete_low_count_ngrams
from src.load_dataset import load_bookcorpus_dataset
from src.utils import read_total_ngrams_per_size, open_db_connection, get_module_logger, get_file_size_bytes, \
    get_db_path

# TODO: add checkpoints? so we can resume if we were interrupted?
# TODO: I probably want to create some progress bar or something.
#  Running this on a very large dataset can take a week... I might want to give some information on the progress.
logger = get_module_logger(__name__)


def run_pipeline(max_ngram_size: int, min_count_threshold: int, vocab_size: int,
                 ngram_size_to_vocab_percent: dict[int, float], n_samples: int, ngram_count_batch_size: int,
                 n_workers: int, filter_ngram_count_threshold: int, save_dir: Path):
    # TODO: make n_samples optional
    # TODO: make n_workers optional. can pass to all functions None, and the value will be determined in the lowest
    #   possible level
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)
    db_connection = open_db_connection(save_dir)

    dataset = load_bookcorpus_dataset(n_samples, n_workers=n_workers)
    count_ngrams_in_batches(
        tokenized_dataset=dataset,
        save_dir=save_dir,
        ngram_count_batch_size=ngram_count_batch_size,
        n_workers=n_workers,
        max_ngram_size=max_ngram_size,
        filter_ngram_count_threshold=filter_ngram_count_threshold,
    )
    aggregate_batch_ngram_counts(save_dir, max_ngram_size, db_connection)
    delete_low_count_ngrams(max_ngram_size, db_connection, min_count_threshold)
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, max_ngram_size)
    compute_pmi_score(db_connection, max_ngram_size)

    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, vocab_size, ngram_size_to_vocab_percent)

    db_connection.close()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db_size_bytes: {db_size_bytes}')

    # TODO: do I want to delete all the data and just return the vocab?
    # TODO: clean up the db_file? if I do that, I will not be able to run the end_to_end_test. maybe add a flag for doing that.
    return pmi_masking_vocab


def run_pipeline_with_experiment_config(experiment_config):
    logger.info(f'start experiment_config: {experiment_config.__name__}')
    pmi_masking_vocab = run_pipeline(
        max_ngram_size=experiment_config.max_ngram_size,
        min_count_threshold=experiment_config.min_count_threshold,
        vocab_size=experiment_config.vocab_size,
        ngram_size_to_vocab_percent=experiment_config.ngram_size_to_vocab_percent,
        n_samples=experiment_config.n_samples,
        ngram_count_batch_size=experiment_config.ngram_count_batch_size,
        n_workers=experiment_config.n_workers,
        filter_ngram_count_threshold=experiment_config.filter_ngram_count_threshold,
        save_dir=experiment_config.save_dir,
    )
    logger.info(f'end experiment_config: {experiment_config.__name__}')
    # TODO: save the vocab to a file (in the previous function?)
    return pmi_masking_vocab


if __name__ == '__main__':
    from experiment_config import medium_size_bookcorpus
    run_pipeline_with_experiment_config(medium_size_bookcorpus)
