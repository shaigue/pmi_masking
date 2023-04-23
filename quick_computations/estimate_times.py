import math

bookcorpus_n_samples = 74_004_228
batch_size = 72_000
subset_size = batch_size * 10
subset_time_sec = 4 * 60
estimated_time_sec = subset_time_sec * (bookcorpus_n_samples / subset_size)
estimated_time_hour = estimated_time_sec / 3600
print(f'estimated time in hours to go over bookcorpus: {estimated_time_hour}.')
n_batches = math.ceil(bookcorpus_n_samples / batch_size)
print(f'There will be {n_batches} batches.')
