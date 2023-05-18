from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(tokenizer_name: str = 'default') -> PreTrainedTokenizerBase:
    # TODO: add support for the desired tokenizer
    if tokenizer_name == 'default':
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        raise NotImplementedError
