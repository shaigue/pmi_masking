from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer() -> PreTrainedTokenizerBase:
    # TODO: figure out what is the correct tokenizer to use
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
