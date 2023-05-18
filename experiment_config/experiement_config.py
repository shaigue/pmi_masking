from dataclasses import dataclass, field

from src.utils import get_cpu_count


@dataclass
class ExperimentConfig:
    """Object representing the experiment configuration.

    Attributes:
        name: configuration name
        dataset_name: name of the dataset to use
        n_samples: number of samples to take from the dataset. If None, uses the entire dataset
        tokenizer_name: name of the tokenizer to use
        max_ngram_size: maximal size of the ngram to consider
        min_count_threshold: after aggregating the counts over the entire dataset, 
            filter ngrams with counts lower than this value.
        vocab_size: size of the resulting pmi masking vocabulary
        ngram_size_to_vocab_percent: percent of each ngram size to put in the final masking vocabulary
        ngram_count_batch_size: batch size for counting ngrams
        n_workers: number of processes to use
        min_count_batch_threshold: filters ngrams in batch counts that appear less than this value
    """
    @staticmethod
    def default_ngram_size_to_vocab_percent() -> dict[int, float]:
        return {2: 50, 3: 25, 4: 12.5, 5: 12.5}

    name: str
    # data parameters
    dataset_name: str = 'bookcorpus'
    n_samples: int = None
    tokenizer_name: str = 'default'
    # TODO: add parameter to randomly sample the dataset.
    # pmi_masking parameters
    max_ngram_size: int = 5
    min_count_threshold: int = 10
    vocab_size: int = 800_000
    ngram_size_to_vocab_percent: dict[int, float] = field(default_factory=default_ngram_size_to_vocab_percent)
    # specific implementation parameters
    ngram_count_batch_size: int = 1_000_000
    n_workers: int = get_cpu_count()
    min_count_batch_threshold: int = 1
