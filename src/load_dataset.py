"""Module responsible for dataset loading"""
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from src.load_tokenizer import load_tokenizer


def load_bookcorpus_dataset() -> Dataset:
    dataset_name = 'bookcorpus'
    split = 'train'
    return load_dataset(dataset_name, split=split)


def load_wikipedia_dataset() -> Dataset:
    dataset_path = 'wikipedia'
    configuration_name = '20220301.en'
    split = 'train'
    return load_dataset(dataset_path, configuration_name, split=split)


def load_bookcorpus_and_wikipedia_dataset():
    # TODO: this should be tested on a large enough machine that has enough disk space
    bookcorpus = load_bookcorpus_dataset()
    wiki = load_wikipedia_dataset()
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
    assert bookcorpus.features.type == wiki.features.type
    dataset = concatenate_datasets([bookcorpus, wiki])
    return dataset


def get_dataset_name_to_load_function() -> dict:
    return {
        'bookcorpus': load_bookcorpus_dataset,
        'wikipedia': load_wikipedia_dataset,
        'bookcorpus+wikipedia': load_bookcorpus_and_wikipedia_dataset,
    }


def get_supported_dataset_names() -> list[str]:
    return list(get_dataset_name_to_load_function().keys())


def tokenize_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, tokenizer_batch_size: int) -> Dataset:
    """Tokenizes a dataset. The tokens will be added to a new column named 'input_ids'.
    :param dataset: dataset to tokenize.
    :param tokenizer: tokenizer to be used.
    :param tokenizer_batch_size: batch size to use for the tokenizer.
    :return: the tokenized dataset.
    """
    def tokenize(batch: dict[str, list]):
        return tokenizer(batch['text'], add_special_tokens=False)

    # Note: since tokenizers use multiprocessing for batches, `num_proc` is not passed to dataset.map()
    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
    )
    return dataset


def load_and_tokenize_dataset(dataset_name: str, tokenizer_name: str, tokenizer_batch_size: int,
                              n_samples: int = None) -> tuple[Dataset, PreTrainedTokenizerBase]:
    """Loads a dataset and tokenizes it.

    :param dataset_name: experiment_name of dataset to load.
    :param tokenizer_name: experiment_name of tokenizer to use.
    :param n_samples: takes the first n_samples of the dataset. If None, uses the entire dataset.
    :param tokenizer_batch_size: batch size for the tokenizer.
    :return: The tokenized dataset.
    """
    dataset_name_to_load_function = get_dataset_name_to_load_function()
    if dataset_name not in dataset_name_to_load_function:
        raise NotImplementedError

    dataset = dataset_name_to_load_function[dataset_name]()

    # TODO: add feature to take random samples of the dataset
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    tokenizer = load_tokenizer(tokenizer_name, dataset=dataset, tokenizer_batch_size=tokenizer_batch_size)
    dataset = tokenize_dataset(dataset, tokenizer, tokenizer_batch_size)

    return dataset, tokenizer
