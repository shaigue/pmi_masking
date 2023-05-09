"""In this module we compare two masking vocabulary to understand their differences"""


# TODO: compare the generated vocabulary to the official masking vocabulary (wiki+bookcorpus)
# TODO: this should probably be refactored and moved somewhere else
def load_vocab_from_file(file: str) -> list[str]:
    with open(file, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    return vocab


def compute_intersection_percent(vocab1: list, vocab2: set) -> tuple[float, float]:
    """Returns the percentage of each vocabulary that is contained in the other one."""
    vocab1_set = set(vocab1)
    vocab2_set = set(vocab2)
    intersection_size = len(vocab1_set.intersection(vocab2_set))
    vocab1_intersection_percent = intersection_size / len(vocab1)
    vocab2_intersection_percent = intersection_size / len(vocab2)
    return vocab1_intersection_percent, vocab2_intersection_percent


def compare_masking_vocabs(vocab_file1: str, vocab_file2: str) -> None:
    """Prints information about the similarity between two masking vocabularies"""
    vocab1 = load_vocab_from_file(vocab_file1)
    vocab2 = load_vocab_from_file(vocab_file2)
    vocab1_intersection_percent, vocab2_intersection_percent = compute_intersection_percent(vocab1, vocab2)
    print(f'intersection percent of {vocab_file1}: {vocab1_intersection_percent}')
    print(f'intersection percent of {vocab_file2}: {vocab2_intersection_percent}')


def compare_official_vocabs():
    compare_masking_vocabs('masking_vocabs/pmi-owt-wiki-bc.txt', 'masking_vocabs/pmi-wiki-bc.txt')


if __name__ == '__main__':
    compare_official_vocabs()

