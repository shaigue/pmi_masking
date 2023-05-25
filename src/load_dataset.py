from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from src.get_tokenizer import get_tokenizer

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


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase,
                     n_workers: int, tokenizer_batch_size: int) -> Dataset:
    """Tokenizes a dataset. The tokens will be added to a new column named 'input_ids'.

    :param dataset: dataset to tokenize.
    :param tokenizer: tokenizer to be used.
    :param n_workers: number of workers to use.
    :param tokenizer_batch_size: batch size to use for the tokenizer.
    :return: the tokenized dataset.
    """
    def tokenize(batch: dict[str, list]):
        return tokenizer(batch['text'], add_special_tokens=False)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=n_workers,
    )
    return dataset


def load_and_tokenize_dataset(dataset_name: str, tokenizer_name: str, n_workers: int, n_samples: int = None,
                              tokenizer_batch_size: int = 4_000) -> Dataset:
    """Loads a dataset and tokenizes it.

    :param dataset_name: name of dataset to load.
    :param tokenizer_name: name of tokenizer to use.
    :param n_workers: number of workers to use.
    :param n_samples: takes the first n_samples of the dataset. If None, uses the entire dataset.
    :param tokenizer_batch_size: batch size for the tokenizer.
    :return: The tokenized dataset.
    """
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

    # TODO: add feature to take random samples of the dataset
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    tokenizer = get_tokenizer(tokenizer_name)
    dataset = tokenize_dataset(dataset, tokenizer, n_workers, tokenizer_batch_size)

    return dataset
