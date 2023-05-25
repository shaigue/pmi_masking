"""Configuration to generate PMI masking vocabulary for the Bookcorpus + Wikipedia datasets"""
from src.experiment_config import ExperimentConfig

config = ExperimentConfig(
    name='bookcorpus_and_wikipedia',
    dataset_name='bookcorpus+wikipedia',
    min_count_batch_threshold=2
)
