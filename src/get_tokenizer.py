from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
