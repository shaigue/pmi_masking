from datasets import load_dataset, Dataset

from src.get_tokenizer import get_tokenizer
from src.utils import tokenize_dataset


def load_bookcorpus_dataset(n_samples: int = None, tokenize: bool = True, n_workers: int = 1,
                            tokenizer_batch_size: int = 4_000) -> Dataset:
    dataset_name = 'bookcorpus'
    split = 'train'
    dataset = load_dataset(dataset_name, split=split)

    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    tokenizer = get_tokenizer()
    if tokenize:
        dataset = tokenize_dataset(dataset, tokenizer, n_workers, tokenizer_batch_size)

    return dataset


def load_wikipedia_dataset() -> Dataset:
    # TODO: when loading a dataset, add functionality like the above `load_bookcorpus_dataset`
    dataset_path = 'wikipedia'
    configuration_name = '20220301.en'
    # the wikipedia dataset cannot be streamed.
    dataset = load_dataset(dataset_path, configuration_name)
    return dataset
