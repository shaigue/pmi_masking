import shutil
from pathlib import Path

from src.compute_log_likelihood import compute_log_likelihood
from src.aggregate_batch_ngram_counts import aggregate_batch_ngram_counts
from src.compute_max_segmentation_log_likelihood_sum import compute_max_segmentation_log_likelihood_sum
from src.compute_pmi_masking_vocab import compute_pmi_masking_vocab
from src.compute_pmi_score import compute_pmi_score
from src.count_ngrams_in_batches import count_ngrams_in_batches
from src.load_dataset import load_bookcorpus_dataset
from src.utils import read_total_ngrams_per_size, open_db_connection


# TODO: add checkpoints? so we can resume if we were interrupted?
# TODO: I probably want to create some progress bar or something.
#  Running this on a very large dataset can take a week... I might want to give some information on the progress.

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
    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)
    compute_max_segmentation_log_likelihood_sum(db_connection, max_ngram_size)
    compute_pmi_score(db_connection, max_ngram_size)
    pmi_masking_vocab = compute_pmi_masking_vocab(db_connection, vocab_size, min_count_threshold,
                                                  ngram_size_to_vocab_percent)

    db_connection.close()
    # TODO: do I want to delete all the data and just return the vocab?

    return pmi_masking_vocab


def run_pipeline_with_parameters(parameters):
    return run_pipeline(
        max_ngram_size=parameters.max_ngram_size,
        min_count_threshold=parameters.min_count_threshold,
        vocab_size=parameters.vocab_size,
        ngram_size_to_vocab_percent=parameters.ngram_size_to_vocab_percent,
        n_samples=parameters.n_samples,
        ngram_count_batch_size=parameters.ngram_count_batch_size,
        n_workers=parameters.n_workers,
        filter_ngram_count_threshold=parameters.filter_ngram_count_threshold,
        save_dir=parameters.save_dir,
    )


if __name__ == '__main__':
    from experiment_parameters import medium_size_bookcorpus_parameters
    run_pipeline_with_parameters(medium_size_bookcorpus_parameters)
