from src.experiment_config import ExperimentConfig

config = ExperimentConfig(
    name='end_to_end_test',
    n_samples=10_000,
    min_count_threshold=5,
    vocab_size=1_000,
    ngram_count_batch_size=1_000,
    n_workers=1,
)
