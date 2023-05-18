from experiment_config.experiement_config import ExperimentConfig

config = ExperimentConfig(
    name='medium_size_bookcorpus',
    n_samples=30_000_000,
    min_count_batch_threshold=2,
)
