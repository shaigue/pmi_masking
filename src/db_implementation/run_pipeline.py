import shutil

from src.experiment_config import ExperimentConfig
from src.db_implementation.compute_log_likelihood import compute_log_likelihood
from src.db_implementation.aggregate_ngram_counts import aggregate_ngram_counts
from src.db_implementation.compute_max_segmentation_log_likelihood_sum import \
    compute_max_segmentation_log_likelihood_sum
from src.db_implementation.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.db_implementation.compute_pmi_score import compute_pmi_score
from src.db_implementation.count_ngrams_in_batches import count_ngrams_in_batches
from src.db_implementation.prune_low_count_ngrams import prune_low_count_ngrams
from src.get_tokenizer import get_tokenizer
from src.load_dataset import load_and_tokenize_dataset
from src.utils import get_module_logger, get_file_size_bytes, Ngram
from src.db_implementation.utils import read_total_ngrams_per_size, get_db_path, open_db_connection, get_save_dir, \
    get_vocab_file

logger = get_module_logger(__name__)


def run_pipeline(experiment_config: ExperimentConfig, clean_up: bool = True,
                 save_vocab_to_file: bool = True) -> list[Ngram]:
    """Runs the entire pipeline for creating the PMI masking vocabulary from a dataset, using a database.

    :param experiment_config: configurations for the experiment, like which dataset to use, tokenizer, ...
    :param clean_up: if True, the database will be deleted after the pmi masking vocabulary is generated.
        otherwise, it will not be deleted.
    :param save_vocab_to_file: if True, the resulting PMI masking vocabulary will be saved to file, after converting
        back the token-ids to strings. Otherwise, it will not be saved.
    :return: a list of the generated pmi masking vocabulary (ngrams are represented with token ids).
    """
    # TODO: add checkpoints? so we can resume if we were interrupted?
    # TODO: add some way to show progress. running this on a large dataset can take a lot of time.
    logger.info(f'start experiment: {experiment_config.name}')
    logger.info(f'n_workers={experiment_config.n_workers}')

    save_dir = get_save_dir(experiment_config.name)
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)
    db_connection = open_db_connection(save_dir)

    dataset = load_and_tokenize_dataset(dataset_name=experiment_config.dataset_name,
                                        tokenizer_name=experiment_config.tokenizer_name, n_workers=experiment_config.n_workers,
                                        n_samples=experiment_config.n_samples)
    count_ngrams_in_batches(
        tokenized_dataset=dataset,
        save_dir=save_dir,
        ngram_count_batch_size=experiment_config.ngram_count_batch_size,
        n_workers=experiment_config.n_workers,
        max_ngram_size=experiment_config.max_ngram_size,
        filter_ngram_count_threshold=experiment_config.min_count_batch_threshold,
    )
    aggregate_ngram_counts(save_dir, db_connection, experiment_config.max_ngram_size)

    db_connection.commit()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after aggregate counts: {db_size_bytes}')

    prune_low_count_ngrams(db_connection, experiment_config.max_ngram_size, experiment_config.min_count_threshold)
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, experiment_config.max_ngram_size)
    compute_pmi_score(db_connection, experiment_config.max_ngram_size)

    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, experiment_config.vocab_size,
                                                  experiment_config.ngram_size_to_vocab_percent)

    db_connection.close()
    db_size_bytes = get_file_size_bytes(get_db_path(save_dir))
    logger.info(f'db size bytes after pmi compute: {db_size_bytes}')

    if clean_up:
        shutil.rmtree(save_dir, ignore_errors=True)

    if save_vocab_to_file:
        tokenizer = get_tokenizer(experiment_config.tokenizer_name)
        lines = tokenizer.batch_decode(pmi_masking_vocab)
        vocab_file = get_vocab_file(experiment_config)
        with vocab_file.open('w') as f:
            for line in lines:
                line += '\n'
                f.write(line)

    logger.info(f'end experiment: {experiment_config.name}')

    return pmi_masking_vocab
