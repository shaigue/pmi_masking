# PMI Masking
Python implementation of the procedure for creating PMI masking vocabulary based 
on the [paper](https://arxiv.org/abs/2010.01825) by AI21 Labs.

The main problem that arises while computing PMI masking vocabulary for large 
datasets is the large number of ngrams, which results in large memory requirements.
In order to process the larger-than-RAM ngram data, 
we use [DuckDB](https://duckdb.org/) in our implementation. 


## Instructions

This project requires Python version 3.9. 
Older versions produce errors because syntax that was introduced only in version 3.9 is used (specific type hints),
and newer versions are incompatible with `apache-beam` package used for loading the 
wikipedia dataset https://github.com/huggingface/datasets/issues/5613.

### Setup
#### Clone the repo
- Enter the directory where you want to clone the repository to
- Clone the repo
- CD into the repo's directory

#### Environment

Note that you might need to update apt-get, install pip and venv before that (might need `sudo` permissions):
```commandline
apt-get update
```
```commandline
apt install python3-pip
```
```commandline
apt install python3-venv
```

Make sure that pip is up-to-date:
```commandline
python3 -m pip install --upgrade pip
```

Create a new virtual environment:
```commandline
python3 -m venv env
```

Activate the virtual environment:
- Linux:
```commandline
source env/bin/activate
```
(`activate` script might be in a different directory, named `Scripts` instead of `bin`)

- Windows:
```commandline
.\env\Scripts\activate
```

Install dependencies from the `requirements.txt` file:
```commandline
python3 -m pip install -r requirements.txt
```

If you wish to use the environment in a jupyter notebook, 
you should install an IPython kernel:
```commandline
python3 -m ipykernel install --user --name pmi_masking --display-name "Python (pmi_masking)"
```

#### Run tests
To verify sure that the setup is successful, run tests (all tests should pass): 
```commandline
python3 -m unittest discover -s tests 
```

### Running
To run the program and create a PMI masking vocabulary,
use the script [create_pmi_masking_vocab.py](create_pmi_masking_vocab.py).
Running the script with the `--help` flag gives information on the 
arguments and how to run it:

```commandline
usage: create_pmi_masking_vocab.py [-h] --experiment_name EXPERIMENT_NAME
                                   --dataset_name
                                   {bookcorpus,wikipedia,bookcorpus+wikipedia}
                                   [--tokenizer_name {bert-base-uncased}]
                                   [--max_ngram_size MAX_NGRAM_SIZE]
                                   [--min_count_threshold MIN_COUNT_THRESHOLD]
                                   [--vocab_size VOCAB_SIZE]
                                   [--ngram_size_to_vocab_percent NGRAM_SIZE_TO_VOCAB_PERCENT [NGRAM_SIZE_TO_VOCAB_PERCENT ...]]
                                   [--ngram_count_batch_size NGRAM_COUNT_BATCH_SIZE]
                                   [--min_count_batch_threshold MIN_COUNT_BATCH_THRESHOLD]
                                   [--n_workers N_WORKERS]
                                   [--tokenizer_batch_size TOKENIZER_BATCH_SIZE]
                                   [--n_samples N_SAMPLES]

Main script for this project. Creates a PMI-masking vocabulary for a dataset.
Resulting vocabulary is saved as text file named `<experiment_name>.txt` in
the directory `pmi_masking_vocabs`. Each line is an n-gram in the PMI-masking
vocabulary. Only supports datasets specified in the `dataset_name` argument.
To add support for other datasets, write a function that loads the dataset in
the file `src/load_dataset.py` and add an entry with the new dataset name as
the key to the dictionary returned by the function
`get_dataset_name_to_load_function()` in that file. Support is automatically
added to this script. Only supports tokenizers specified in the
`tokenizer_name` argument. The process for adding a tokenizer is similar to
adding a dataset. To add support for other tokenizers, write a function that
loads the tokenizer in the file `src/load_tokenizer.py` and add an entry with
the new tokenizer name as the key to the dictionary returned by the function
`get_tokenizer_name_to_load_function()` in that file. Support is automatically
added to this script.

options:
  -h, --help            show this help message and exit
  --experiment_name EXPERIMENT_NAME
                        experiment experiment_name. affects logging and
                        resulting file names
  --dataset_name {bookcorpus,wikipedia,bookcorpus+wikipedia}
                        determines which dataset to use
  --tokenizer_name {bert-base-uncased}
                        which tokenizer to use
  --max_ngram_size MAX_NGRAM_SIZE
                        maximum ngram size to consider
  --min_count_threshold MIN_COUNT_THRESHOLD
                        prunes ngrams that appear less than this amount in the
                        entire dataset
  --vocab_size VOCAB_SIZE
                        number of ngrams (excluding unigrams) to select for
                        the PMI masking vocabulary
  --ngram_size_to_vocab_percent NGRAM_SIZE_TO_VOCAB_PERCENT [NGRAM_SIZE_TO_VOCAB_PERCENT ...]
                        percentage of ngram size to include in the resulting
                        vocabulary. this should be a list of values, one for
                        each ngram size, from 2 to `max_ngram_size`. for
                        example, `--ngram_size_to_vocab_percent 50 25 12.5
                        12.5` means that the resulting vocabulary will contain
                        50% ngrams of size 2, 25% ngrams of size 3, 12.5%
                        ngrams of size 4 and 12.5% ngrams of size 5. values
                        should sum up to 100% and every ngram should get a
                        positive value
  --ngram_count_batch_size NGRAM_COUNT_BATCH_SIZE
                        ngrams are first counted in batches instead of the
                        entire dataset, for parallelization. this is the
                        number of samples that goes into each batch. if value
                        is too high, counts will not fit into memory and this
                        will slow the program. low values will create a lot of
                        context switches and will also slow down the program
  --min_count_batch_threshold MIN_COUNT_BATCH_THRESHOLD
                        ngrams that occur less than this amount in a batch
                        will be pruned from that batch counts. value of 1
                        means that all the ngrams that appear in a batch will
                        be counted, and value of 2 means that ngrams that
                        appear only once in a batch will be pruned from that
                        batch counts. since most ngrams appear once, using a
                        value >= 2 can greatly reduce space and time
                        requirements
  --n_workers N_WORKERS
                        number of workers to use. defaults to the number of
                        available CPUs
  --tokenizer_batch_size TOKENIZER_BATCH_SIZE
                        batch size for the tokenization step
  --n_samples N_SAMPLES
                        if provided, only the first `n_samples` samples of the
                        dataset will be used. if not, the entire dataset will
                        be used. This argument is for testing and
                        experimentation purposes
```

Note that only a limited set of tokenizers and 
datasets are supported. Instructions on how to add support for new tokenizers/datasets appear
at the beginning of the help message.

#### Logging
Running the program on a large dataset might take
a while.
Logging messages will be printed to console and to the 
file `log.log`.
Use this logs to measure progress.

#### Program stages
The stages of the programs are:
1. `count_ngrams_in_batches` - splits the dataset into batches and counts ngrams 
in each batch.
2. `aggregate_ngram_counts` - aggregates the counts from the batches into a single database.
this step takes the longest.
3. `prune_low_count_ngrams` - prunes ngrams that occur in the dataset less than a given 
number of times.
4. `compute_log_likelihood` - computes the log likelihood scores of the 
ngrams.
5. `compute_max_segmentation_log_likelihood_sum` - computes an intermediate value that will 
be used for computing the PMI scores.
6. `compute_pmi_score` - computes the PMI scores for ngrams
7. `compute_pmi_masking_vocab` - takes the ngrams with the highest PMI scores
and creates the PMI-masking vocabulary.

## Performance and resource requirements
In this section we present performance results on different datasets and systems. 
You cna use those numbers for a rough estimate how much resources it will take for your setting.

|  dataset   |     #tokens     |                      processor                      | #processors |    memory    |                     system                      | total time | disk space |
|:----------:|:---------------:|:---------------------------------------------------:|:-----------:|:------------:|:-----------------------------------------------:|:----------:|:----------:|
| bookcorpus |      None       | Intel64 Family 6 Model 142 Stepping 9, GenuineIntel |      4      |   7.88 GB    |            Windows-10-10.0.19045-SP0            | 6.73 hours |  5.13 GB   |
| bookcorpus | 1,098,720,840   |                       x86_64                        |    120      | 1007.59 GB   | Linux-5.4.0-148-generic-x86_64-with-glibc2.31   | 4.45 hours |  3.98 GB   |

### Datasets

* [bookcorpus](https://huggingface.co/datasets/bookcorpus)
* [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
* [wikipedia 20220301en](https://huggingface.co/datasets/wikipedia#20220301en)
