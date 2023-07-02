"""Module responsible for loading the tokenizer"""
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

from src.utils import get_module_logger

logger = get_module_logger(__name__)

__all__ = [
    'get_supported_tokenizer_names',
    'load_tokenizer'
]


def load_bert_base_uncased_tokenizer(*args, **kwargs) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained('bert-base-uncased')


def load_word_level_tokenizer(dataset: Dataset, tokenizer_batch_size: int, *args, **kwargs) -> PreTrainedTokenizerBase:
    tokenizer_ = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer_.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer_.normalizer = normalizers.BertNormalizer()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    # TODO: maybe add vocab_size as an input parameter?
    trainer = trainers.WordLevelTrainer(vocab_size=500_000, special_tokens=special_tokens, show_progress=True)

    def batch_iterator(batch_size: int):
        for batch in dataset.iter(batch_size=batch_size):
            yield batch["text"]

    logger.info('start training word level tokenizer')
    tokenizer_.train_from_iterator(batch_iterator(tokenizer_batch_size), trainer=trainer, length=len(dataset))
    logger.info('end training word level tokenizer')

    return PreTrainedTokenizerFast(tokenizer_object=tokenizer_)


def get_tokenizer_name_to_load_function() -> dict:
    return {
        'bert-base-uncased': load_bert_base_uncased_tokenizer,
        'word-level': load_word_level_tokenizer,
    }


def get_supported_tokenizer_names() -> list[str]:
    return list(get_tokenizer_name_to_load_function().keys())


def load_tokenizer(tokenizer_name: str, dataset: Dataset, tokenizer_batch_size: int) -> PreTrainedTokenizerBase:
    tokenizer_name_to_load_function = get_tokenizer_name_to_load_function()
    if tokenizer_name not in tokenizer_name_to_load_function:
        raise NotImplementedError

    return tokenizer_name_to_load_function[tokenizer_name](dataset, tokenizer_batch_size)
