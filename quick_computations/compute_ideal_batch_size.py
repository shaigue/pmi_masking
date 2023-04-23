original_batch_size = 4_000
counter_size_in_mb = 10
available_memory_in_mb = 1_800
percentage_of_memory = 0.1  # 10%

target_size_in_mb = available_memory_in_mb * percentage_of_memory
counter_size_in_mb_per_sample = counter_size_in_mb / original_batch_size
ideal_batch_size = round(target_size_in_mb / counter_size_in_mb_per_sample)
print(f'ideal batch size: {ideal_batch_size}')
