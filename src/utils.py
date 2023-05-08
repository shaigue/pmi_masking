# TODO: move here functions that are used in multiple modules. I don't want
#   the different modules to be inter-dependent
import json
import logging
from enum import Enum, auto, StrEnum
from pathlib import Path

from datasets import Dataset as HuggingfaceDataset
from transformers import PreTrainedTokenizerBase


Ngram = tuple[int, ...]


# TODO: use this enum all over the place...
class Field(StrEnum):
    count = auto()
    log_likelihood = auto()
    max_segmentation_log_likelihood_sum = auto()
    pmi_score = auto()


def get_module_logger(name: str) -> logging.Logger:
    """
    :param name: should be __name__ special variable in the module that calls this
        function.
    """
    # TODO: maybe add some config file to the directory for project
    #  level configurations.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logs_dir = Path(__file__).parents[1] / 'logs'
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / 'log.log'
    handler = logging.FileHandler(str(log_file), mode='a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_token_field_name(token_i: int) -> str:
    return f'token_{token_i}'


def get_token_field_sql_type() -> str:
    return 'UINTEGER'


def get_count_field_sql_type() -> str:
    return 'UINTEGER'


def get_token_field_declaration_str(token_i: int) -> str:
    return f'{get_token_field_name(token_i)} {get_token_field_sql_type()}'


def get_count_field_declaration_str() -> str:
    return f'count {get_count_field_sql_type()}'


def get_ngram_table_name(ngram_size: int) -> str:
    return f'ngram_of_size_{ngram_size}_counts_table'


def get_token_field_names(ngram_size: int) -> list[str]:
    return [get_token_field_name(token_i) for token_i in range(ngram_size)]


def get_key_str(ngram_size: int) -> str:
    return ', '.join(get_token_field_name(token_i) for token_i in range(ngram_size))


def get_total_ngrams_per_size_file(save_dir):
    return save_dir / 'total_ngrams_per_size.json'




def read_total_ngrams_per_size(save_dir: Path) -> dict[int, int]:
    json_file = get_total_ngrams_per_size_file(save_dir)
    with json_file.open('r') as f:
        total_ngrams_per_size = json.load(f)
    # TODO convert keys to integers
    return total_ngrams_per_size


def tokenize_dataset(dataset: HuggingfaceDataset, tokenizer: PreTrainedTokenizerBase,
                     n_workers: int, tokenizer_batch_size: int):
    def tokenize(batch: dict[str, list]):
        return tokenizer(batch['text'], add_special_tokens=False)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenizer_batch_size,
        num_proc=n_workers,
    )
    return dataset
