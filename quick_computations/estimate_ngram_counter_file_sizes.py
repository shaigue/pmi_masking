import os
from pathlib import Path
from statistics import mean
# TODO: think about how to organize those scripts?

n_samples_in_dataset = 74_004_228
batch_size = 72_000

counter_dir = Path.cwd().absolute().parent / 'counters'
counter_file_size = [
    os.path.getsize(ngram_counter_file)
    for ngram_counter_file in counter_dir.glob('ngram_counter_*.json')
]
print(f'counter file sizes: {counter_file_size}')

n_batches = n_samples_in_dataset // batch_size
print(f'expected number of files: {n_batches}')

average_size_in_bytes = mean(counter_file_size)
expected_size_in_bytes = n_batches * average_size_in_bytes
expected_size_in_giga_bytes = expected_size_in_bytes / (2 ** 30)
print(f'expected size in giga bytes: {expected_size_in_giga_bytes}')
