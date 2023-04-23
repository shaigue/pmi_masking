"""This module is for loading the datasets for constructing the PMI masking vocabulary."""
import time
from collections.abc import Iterable
from typing import Union

# TODO: Ideally, I would like to measure how much progress I have made, but I don't see how
#   to do that with IterableDataset (since we want to stream the dataset.). Maybe Zach can help.
# TODO: maybe, in order to measure progress, I could first just iterate through the en
# TODO: Load the wikipedia dataset.

from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer() -> PreTrainedTokenizerBase:
    # TODO: figure out what is the correct tokenizer to use
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer


def __process_dataset(dataset: Union[Dataset, IterableDataset],
                      tokenizer: PreTrainedTokenizerBase, batch_size: int,
                      n_examples: int = None) -> Iterable[list[list[int]]]:
    if n_examples is not None:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(n_examples)
        elif isinstance(dataset, Dataset):
            # TODO: check if this returns a dataset.
            dataset = dataset.select(range(n_examples))
        else:
            raise TypeError(f'expected type {Union[Dataset, IterableDataset]}, '
                            f'got {type(dataset)}.')
    # TODO: I'm getting a DatasetDict with wikipedia, So I need to figure out how to deal with that.
    dataset = dataset.iter(batch_size)

    def tokenize_batch(batch: dict[str, list]) -> list[list[int]]:
        return tokenizer(batch['text'], add_special_tokens=False)['input_ids']

    dataset = map(tokenize_batch, dataset)
    return dataset


def load_and_tokenize_bookcorpus_dataset(tokenizer: PreTrainedTokenizerBase, batch_size: int = 1_000,
                                         n_examples: int = None) -> Iterable[list[list[int]]]:
    dataset_name = 'bookcorpus'
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    return __process_dataset(dataset, tokenizer, batch_size, n_examples)


def load_and_tokenize_wikipedia_dataset(tokenizer: PreTrainedTokenizerBase, batch_size: int = 1_000,
                                        n_examples: int = None) -> Iterable[list[list[int]]]:
    dataset_path = 'wikipedia'
    configuration_name = '20220301.en'
    # the wikipedia dataset cannot be streamed.
    dataset = load_dataset(dataset_path, configuration_name)
    return __process_dataset(dataset, tokenizer, batch_size, n_examples)


def __iterate_entire_wikipedia_english_dataset():
    """iterate the entire dataset to figure out how many examples are there"""
    print('Iterating over the entire dataset...')
    tokenizer = get_tokenizer()
    batch_size = 1_000
    dataset = load_and_tokenize_wikipedia_dataset(tokenizer)
    t1 = time.time()
    i = 0
    for example in dataset:
        i += 1
        if i % 100_000 == 0:
            print(f'Example number {i * batch_size}.')
    t2 = time.time()
    print(f'Iterating over the entire dataset took: {round(t2 - t1, 4)} seconds.')
    print(f'Number of examples in the dataset: {i * batch_size}.')


if __name__ == '__main__':
    # __iterate_entire_bookcorpus_dataset()
    __iterate_entire_wikipedia_english_dataset()
