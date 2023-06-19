"""Module responsible for loading the tokenizer"""
from transformers import AutoTokenizer, PreTrainedTokenizerBase


__all__ = [
    'get_supported_tokenizer_names',
    'load_tokenizer'
]


def load_bert_base_uncased_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def get_tokenizer_name_to_load_function() -> dict:
    return {
        'bert-base-uncased': load_bert_base_uncased_tokenizer,
    }


def get_supported_tokenizer_names() -> list[str]:
    return list(get_tokenizer_name_to_load_function().keys())


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    tokenizer_name_to_load_function = get_tokenizer_name_to_load_function()
    if tokenizer_name not in tokenizer_name_to_load_function:
        raise NotImplementedError

    return tokenizer_name_to_load_function[tokenizer_name]()