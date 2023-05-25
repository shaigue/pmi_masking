"""Configuration for generating PMI vocabulary for the bookcorpus dataset"""
from src.experiment_config import ExperimentConfig

config = ExperimentConfig(
    name='bookcorpus',
    min_count_batch_threshold=2,
)
