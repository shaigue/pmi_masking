import shutil
from pathlib import Path

import duckdb

from src.compute_log_likelihood import compute_log_likelihood
from src.aggregate_batch_ngram_counts import aggregate_batch_ngram_counts
from src.compute_max_segmentation_log_likelihood_sum import compute_max_segmentation_log_likelihood_sum
from src.count_ngrams_in_batches import count_ngrams_in_batches
from src.get_tokenizer import get_tokenizer
from src.load_dataset import load_bookcorpus_dataset
from src.utils import read_total_ngrams_per_size

if __name__ == '__main__':
    n_samples = 30_000_000
    ngram_count_batch_size = 1_000_000
    n_workers = 3
    max_ngram_size = 5
    filter_ngram_count_threshold = 2

    save_dir = Path('./data')
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(exist_ok=True)

    dataset = load_bookcorpus_dataset()
    dataset = dataset.select(range(n_samples))

    tokenizer = get_tokenizer()

    count_ngrams_in_batches(
        dataset=dataset,
        tokenizer=tokenizer,
        save_dir=save_dir,
        ngram_count_batch_size=ngram_count_batch_size,
        n_workers=n_workers,
        max_ngram_size=max_ngram_size,
        filter_ngram_count_threshold=filter_ngram_count_threshold,
    )

    database_file = save_dir / 'ngram_data.duckdb'
    db_connection = duckdb.connect(str(database_file))
    aggregate_batch_ngram_counts(save_dir, max_ngram_size, db_connection)

    total_ngrams_per_size = read_total_ngrams_per_size(save_dir)
    compute_log_likelihood(db_connection, total_ngrams_per_size)

    compute_max_segmentation_log_likelihood_sum(db_connection, max_ngram_size)

    db_connection.close()
