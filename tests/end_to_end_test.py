"""n end-to-end test comparing the naive implementation and the DuckDB based implementation"""
import shutil
from pathlib import Path

import duckdb

from src.aggregate_batch_ngram_counts import aggregate_batch_ngram_counts
from src.count_ngrams_in_batches import count_ngrams_in_batches
from src.get_tokenizer import get_tokenizer
from src.load_dataset import load_bookcorpus_dataset
from src.naive_implementation import count_ngrams_from_dataset
from src.utils import get_ngram_table_name, get_token_field_name


def collect_ngram_counts_from_db(database_file: Path, max_ngram_size: int) -> dict[tuple[int, ...], int]:
    connection = duckdb.connect(str(database_file))
    ngram_counts = {}
    for ngram_size in range(1, max_ngram_size + 1):
        table_name = get_ngram_table_name(ngram_size)
        table = connection.sql(f'SELECT * FROM {table_name}').arrow()
        for row in table.to_pylist():
            ngram = tuple(row[get_token_field_name(token_i)] for token_i in range(ngram_size))
            ngram_counts[ngram] = row['count']
    connection.close()
    return ngram_counts


def end_to_end_test():
    # TODO: test the `count_ngrams_in_batches` and `aggregate_batch_ngram_counts`
    dataset = load_bookcorpus_dataset()
    n_samples = 1_000
    dataset = dataset.select(range(n_samples))
    tokenizer = get_tokenizer()

    save_dir = Path('../end_to_end_test_data')
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir()
    database_file = save_dir / 'ngram_counts.duckdb'
    ngram_count_batch_size = 100
    max_ngram_size = 5

    count_ngrams_in_batches(
        dataset=dataset,
        tokenizer=tokenizer,
        save_dir=save_dir,
        ngram_count_batch_size=ngram_count_batch_size,
        max_ngram_size=max_ngram_size,
    )
    aggregate_batch_ngram_counts(
        save_dir=save_dir,
        max_ngram_size=max_ngram_size,
        db_connection=database_file,
    )
    result = collect_ngram_counts_from_db(database_file, max_ngram_size)
    expected = count_ngrams_from_dataset(dataset, tokenizer, max_ngram_size)
    if expected == result:
        print("Counting ngrams test passed.")
    else:
        print("counting ngrams test failed.")
        print(f"expected:\n{expected}")
        print(f"result:\n{result}")
        return
    # TODO: continue testing the other parts


if __name__ == '__main__':
    end_to_end_test()
