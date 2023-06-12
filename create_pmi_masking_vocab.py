from argparse import ArgumentParser
from importlib import import_module

import config
from src.db_implementation.run_pipeline import run_pipeline
from src.utils import get_module_logger, get_available_cpus_count

logger = get_module_logger(__name__)
# TODO: move all parameters and input validation to here
# TODO: use the help flag on this script, and check if the output is useful
# TODO: delete the experiment config object? or maybe only use it internally?
# TODO: I want to hide the warning when loading transformers about not having pytorch
#   installed.
# TODO: go over the `help` parameters and make sure It is clear.


def get_experiment_config_options() -> list[str]:
    """Returns a list of all the available options for experiment config."""
    experiment_config_dir = config.EXPERIMENT_CONFIG_DIR
    config_name_options = [file.stem for file in experiment_config_dir.iterdir() if file.suffix == '.py']
    return config_name_options


def get_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # TODO: I want all arguments to be full named arguments. I think that
    #   this is the most clear way to write that.
    # TODO: sort out required and not required arguments.
    # TODO: set up default values
    # TODO: make sure to do input validity checks
    # TODO: make sure to add help for each of the arguments
    # TODO: consider using a unified model for the return type
    # TODO: standardize the args somehow and pass to logging & running
    #  I think that simplest way to achieve that is to just pass the arguments
    #  directly, without having a special class for that. the description will
    #  appear in the documentation + in here
    # TODO: make sure to organize the arguments such that required arguments appear first
    # action: store, store_true, store_false, append
    # nargs: N, ?, *, +
    # default:
    # type:
    # choices:
    # required: True/False
    # help
    # TODO: create an input validation function.
    # TODO: When doing input validation, use a separate function, and raise ArgumentError
    #
    parser.add_argument(
        '--experiment_name',
        help='Name of the experiment. Affects logging and resulting file names.',
        type=str,
        required=True
    )
    # TODO: the choices in here should probably come from the load_dataset module, and
    #   not be duplicated here.
    parser.add_argument(
        '--dataset_name',
        help='Which dataset to use.',
        type=str,
        choices=['bookcorpus', 'wikipedia', 'bookcorpus+wikipedia'],
        required=True
    )

    parser.add_argument(
        '--tokenizer_name',
        help='Name of the tokenizer to use.',
        type=str,
        choices=['default'],
        default='default',
    )
    # TODO: validate that this is a positive number, larger than 1.
    parser.add_argument(
        '--max_ngram_size',
        help='Maximal ngram size to consider.',
        type=int,
        default=5,
    )
    # TODO: validate this is a positive number, and is >= from the batch version.
    parser.add_argument(
        '--min_count_threshold',
        help='Prunes ngrams that appear less than this amount in the entire dataset.',
        type=int,
        default=10,
    )
    # TODO: validate a positive number
    parser.add_argument(
        '--vocab_size',
        help='Number of ngrams (excluding unigrams) to select for the PMI masking vocabulary.',
        type=int,
        default=800_000,
    )
    # TODO: figure out what is a good way to pass this type of parameter
    # TODO: validate that it sums up to 100
    # TODO: validate that each ngram size is specified
    # TODO: this should receive a list, where the percent are ordered (first is ngrams
    #  of size 2, size 3, size 4, ...
    # TODO: add example for all of the arguments in the help.
    parser.add_argument(
        '--ngram_size_to_vocab_percent',
        help='Percentage of ngram size to include in the resulting vocabulary. '
             'Expected a list of values, one for each ngram size from 2 to `max_ngram_size`. '
             'For example, `--ngram_size_to_vocab_percent 50 25 12.5 12.5 means '
             'that the resulting vocabulary will contain 50% ngrams of size 2, 25% ngrams of '
             'size 3, 12.5% ngrams of size 4 and 12.5% ngrams of size 5. '
             'Note that values should sum up to 100% and every ngram should get a positive value.',
        nargs='+',
        type=float,
        default=[50., 25., 12.5, 12.5],
    )
    # TODO: validate positive value
    parser.add_argument(
        '--ngram_count_batch_size',
        help='Ngrams are first counted in batches instead of the entire dataset, '
             'for parallelization. '
             'This value specifies how many samples from the dataset goes to each batch. '
             'If the value is too high, counts will not fit to memory and this might slow the program, '
             'low values can also slow the program down, as it will introduce a lot of context switches.',
        type=int,
        default=1_000_000
    )
    # TODO: validate positive number
    parser.add_argument(
        '--n_workers',
        help='Number of workers to use. Defaults to the number of available CPUs.',
        type=int,
        default=get_available_cpus_count()
    )
    # TODO: validate positive value smaller than `min_count_batch`
    parser.add_argument(
        '--min_count_batch_threshold',
        help='Ngrams that occur less than this amount in a batch will be pruned from '
             'this batch counts. Value of 1 means that all the ngrams that appear in a batch '
             'will be counted, and value of 2 means that ngrams that appear only once in a batch '
             'will not be counted in that batch. Since most ngrams appear once, using a value of '
             '2 or greater can greatly reduce space and time requirements.',
        type=int,
        default=1
    )
    # TODO: this argument is for experimentation purposes. move it down.
    # TODO: validate inputs
    parser.add_argument(
        '--n_samples',
        help='If provided, the program will only use the first `n_samples` samples of '
             'the dataset. If not provided, the program will use the entire dataset. '
             'This argument is mainly for testing and experimentation purposes.',
        type=int,
        default=None,
    )
    return parser


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


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    # logger.info(f'start experiment_config: {experiment_config_name}')
    # run_pipeline(experiment_config)
    # logger.info(f'end experiment_config: {experiment_config_name}')


if __name__ == '__main__':
    main()
