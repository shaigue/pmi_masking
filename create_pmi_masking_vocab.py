from argparse import ArgumentParser, Namespace, ArgumentTypeError
from collections.abc import Callable
from importlib import import_module

import config
from src.db_implementation.run_pipeline import run_pipeline
from src.utils import get_module_logger, get_available_cpus_count

logger = get_module_logger(__name__)
# TODO: move all parameters and input validation to here
# TODO: pass arguments down the line explicitly. don't use a special config dataclass. I think this is simplest.
# TODO: delete the experiment config object? or maybe only use it internally?
# TODO: I want to hide the warning when loading transformers about not having pytorch
#   installed.


def get_experiment_config_options() -> list[str]:
    """Returns a list of all the available options for experiment config."""
    experiment_config_dir = config.EXPERIMENT_CONFIG_DIR
    config_name_options = [file.stem for file in experiment_config_dir.iterdir() if file.suffix == '.py']
    return config_name_options


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # methods for input validation.
    def min_value_int_type(min_value: int) -> Callable[[str], int]:
        def min_value_int(s: str) -> int:
            i = int(s)
            if i < min_value:
                raise ArgumentTypeError(f'should be >= {min_value}')
            return i
        return min_value_int
    positive_int = min_value_int_type(1)

    def positive_float(s: str) -> float:
        f = float(s)
        if f <= 0:
            raise ArgumentTypeError(f'should be strictly positive')
        return f

    parser.add_argument(
        '--experiment_name',
        help='experiment name. affects logging and resulting file names',
        type=str,
        required=True
    )
    # TODO: choices here should come from the `load_dataset` module, and not duplicated here.
    parser.add_argument(
        '--dataset_name',
        help='determines which dataset to use',
        type=str,
        choices=['bookcorpus', 'wikipedia', 'bookcorpus+wikipedia'],
        required=True
    )
    # TODO: choices here should come from `load_tokenizer` module, and not duplicated here.
    parser.add_argument(
        '--tokenizer_name',
        help='tokenizer to use',
        type=str,
        choices=['default'],
        default='default',
    )
    parser.add_argument(
        '--max_ngram_size',
        help='maximum ngram size to consider',
        type=min_value_int_type(2),
        default=5,
    )
    parser.add_argument(
        '--min_count_threshold',
        help='prunes ngrams that appear less than this amount in the entire dataset',
        type=positive_int,
        default=10,
    )
    parser.add_argument(
        '--vocab_size',
        help='number of ngrams (excluding unigrams) to select for the PMI masking vocabulary',
        type=positive_int,
        default=800_000,
    )
    # TODO: I changed the format from dictionary to list. make sure down the line
    #   is aware of that.
    parser.add_argument(
        '--ngram_size_to_vocab_percent',
        help='percentage of ngram size to include in the resulting vocabulary. '
             'this should be a list of values, one for each ngram size, from 2 to `max_ngram_size`. '
             'for example, `--ngram_size_to_vocab_percent 50 25 12.5 12.5` means '
             'that the resulting vocabulary will contain 50%% ngrams of size 2, 25%% ngrams of '
             'size 3, 12.5%% ngrams of size 4 and 12.5%% ngrams of size 5. '
             'values should sum up to 100%% and every ngram should get a positive value',
        nargs='+',
        type=positive_float,
        default=[50., 25., 12.5, 12.5],
    )
    parser.add_argument(
        '--ngram_count_batch_size',
        help='ngrams are first counted in batches instead of the entire dataset, '
             'for parallelization. '
             'this is the number of samples that goes into each batch. '
             'if value is too high, counts will not fit into memory and this will slow the program. '
             'low values will create a lot of context switches and will also slow down the program',
        type=positive_int,
        default=1_000_000
    )
    parser.add_argument(
        '--min_count_batch_threshold',
        help='ngrams that occur less than this amount in a batch will be pruned from '
             'that batch counts. value of 1 means that all the ngrams that appear in a batch '
             'will be counted, and value of 2 means that ngrams that appear only once in a batch '
             'will be pruned from that batch counts. since most ngrams appear once, using a value >= 2 '
             'can greatly reduce space and time requirements',
        type=positive_int,
        default=1
    )
    parser.add_argument(
        '--n_workers',
        help='number of workers to use. defaults to the number of available CPUs',
        type=positive_int,
        default=get_available_cpus_count()
    )
    parser.add_argument(
        '--n_samples',
        help='if provided, only the first `n_samples` samples of '
             'the dataset will be used. if not, the entire dataset will be used. '
             'This argument is for testing and experimentation purposes',
        type=positive_int,
        default=None,
    )
    return parser


def validate_arguments(parser: ArgumentParser, args: Namespace) -> None:
    ngram_size_to_vocab_percent = args.ngram_size_to_vocab_percent
    expected_len = args.max_ngram_size - 1
    if len(args.ngram_size_to_vocab_percent) != expected_len:
        parser.error(f'ngram_size_to_vocab_percent should be of length {expected_len}, got length '
                     f'{len(ngram_size_to_vocab_percent)}')
    if sum(ngram_size_to_vocab_percent) != 100:
        parser.error(f'ngram_size_to_vocab_percent should sum to 100. {ngram_size_to_vocab_percent} '
                     f'sums to {sum(ngram_size_to_vocab_percent)}')
    if args.min_count_threshold <= args.min_count_batch_threshold:
        parser.error(f'min_count_threshold should be greater than min_count_batch_threshold. '
                     f'got min_count_threshold={args.min_count_threshold} and '
                     f'min_count_batch_threshold={args.min_count_batch_threshold}')


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    print(args)


def old_main():
    parser = ArgumentParser()
    experiment_config_options = get_experiment_config_options()
    parser.add_argument('--experiment_config', choices=experiment_config_options)
    args = parser.parse_args()
    experiment_config_name = args.experiment_config
    logger.info(f'start experiment_config: {experiment_config_name}')
    experiment_config = import_module(f'experiment_config.{experiment_config_name}').config
    run_pipeline(experiment_config)
    logger.info(f'end experiment_config: {experiment_config_name}')


if __name__ == '__main__':
    main()
