import shutil
from pathlib import Path

from src.aggregate_batch_ngram_counts import aggregate_batch_ngram_counts
from src.count_ngrams_in_batches import count_ngrams_in_batches
from src.get_tokenizer import get_tokenizer
from src.load_dataset import load_bookcorpus_dataset


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
    aggregate_batch_ngram_counts(save_dir, max_ngram_size, database_file)
