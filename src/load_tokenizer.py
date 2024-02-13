"""Module responsible for loading the tokenizer"""
import typing

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

def load_t5_tokenizer(*args, **kwargs) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained('google/t5-v1_1-base')


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
        't5': load_t5_tokenizer,
    }


def get_supported_tokenizer_names() -> typing.List[str]:
    return list(get_tokenizer_name_to_load_function().keys())


def load_tokenizer(tokenizer_name: str, dataset: Dataset, tokenizer_batch_size: int) -> PreTrainedTokenizerBase:
    tokenizer_name_to_load_function = get_tokenizer_name_to_load_function()
    if tokenizer_name not in tokenizer_name_to_load_function:
        raise NotImplementedError(f"Tokenizer {tokenizer_name} is not supported.")

    return tokenizer_name_to_load_function[tokenizer_name](dataset, tokenizer_batch_size)


"""
create_pmi_masking_vocab.py [-h] --experiment_name EXPERIMENT_NAME
                                   --dataset_name
                                   {bookcorpus,wikipedia,bookcorpus+wikipedia}
                                   [--tokenizer_name {bert-base-uncased,word-level}]
                                   [--max_ngram_size MAX_NGRAM_SIZE]
                                   [--min_count_threshold MIN_COUNT_THRESHOLD]
                                   [--vocab_size VOCAB_SIZE]
                                   [--ngram_size_to_vocab_percent NGRAM_SIZE_TO_VOCAB_PERCENT [NGRAM_SIZE_TO_VOCAB_PERCENT ...]]
                                   [--ngram_count_batch_size NGRAM_COUNT_BATCH_SIZE]
                                   [--min_count_batch_threshold MIN_COUNT_BATCH_THRESHOLD]
                                   [--n_workers N_WORKERS]
                                   [--tokenizer_batch_size TOKENIZER_BATCH_SIZE]
                                   [--n_samples N_SAMPLES]
                                   
                                   
python create_pmi_masking_vocab.py \
--experiment_name c4_pmi_masking \
--dataset_name c4 \
--tokenizer_name bert-base-uncased \
--max_ngram_size 5 \
--min_count_threshold 50 \
--vocab_size 50000 \
--ngram_size_to_vocab_percent 15 25 30 30 \
--ngram_count_batch_size 100_000 \
--min_count_batch_threshold 10 \
--n_workers 2 \
--tokenizer_batch_size 1000
"""
