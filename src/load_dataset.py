from datasets import load_dataset, Dataset


def load_bookcorpus_dataset() -> Dataset:
    dataset_name = 'bookcorpus'
    split = 'train'
    return load_dataset(dataset_name, split=split)


def load_wikipedia_dataset() -> Dataset:
    dataset_path = 'wikipedia'
    configuration_name = '20220301.en'
    # the wikipedia dataset cannot be streamed.
    dataset = load_dataset(dataset_path, configuration_name)
    return dataset
