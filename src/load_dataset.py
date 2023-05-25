from datasets import load_dataset, Dataset, concatenate_datasets

from src.get_tokenizer import get_tokenizer
from src.utils import tokenize_dataset
__all__ = ['load_and_tokenize_dataset']


def load_bookcorpus_dataset() -> Dataset:
    dataset_name = 'bookcorpus'
    split = 'train'
    return load_dataset(dataset_name, split=split)


def load_wikipedia_dataset() -> Dataset:
    dataset_path = 'wikipedia'
    configuration_name = '20220301.en'
    split = 'train'
    return load_dataset(dataset_path, configuration_name, split=split)


def load_and_tokenize_dataset(dataset_name: str, tokenizer_name: str, n_workers: int, n_samples: int = None,
                              tokenizer_batch_size: int = 4_000) -> Dataset:
    if dataset_name == 'bookcorpus':
        dataset = load_bookcorpus_dataset()
    elif dataset_name == 'wikipedia':
        dataset = load_wikipedia_dataset()
    elif dataset_name == 'bookcorpus+wikipedia':
        # TODO: this should be tested on a large enough machine that has enough disk space
        bookcorpus = load_bookcorpus_dataset()
        wiki = load_wikipedia_dataset()
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
        assert bookcorpus.features.type == wiki.features.type
        dataset = concatenate_datasets([bookcorpus, wiki])
    else:
        raise NotImplementedError

    # TODO: add ability to random sample the dataset
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    tokenizer = get_tokenizer(tokenizer_name)
    dataset = tokenize_dataset(dataset, tokenizer, n_workers, tokenizer_batch_size)

    return dataset
