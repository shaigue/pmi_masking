# PMI Masking
This repository contains code that takes a text dataset and creates a PMI masking vocabulary for it.

## Overview
Pure Python implementation of the procedure for creating PMI masking vocabulary based 
on this [paper](https://arxiv.org/abs/2010.01825) by AI21 Labs.

The main problem that arises while computing PMI masking vocabulary for large 
datasets is the large number of ngrams, which results in large memory requirements.
In order to process the larger-than-RAM ngram data, 
we use [DuckDB](https://duckdb.org/) in our implementation. 


## How To Run
### Step 1: clone the repo
- Enter the directory where you want to clone the repository to
- Create GitHub SSH keys to enable cloning the repo (https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)
- Clone the repo with SSH
- CD into the repo's directory


### Step 2: Set up environment
Make sure you have the latest pip:
```commandline
python3 -m pip install --upgrade pip
```

Create a new environment:
```commandline
python3 -m venv env
```

Activate the virtual environment:
- Linux:
```commandline
source env/Scripts/activate
```
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


### Step 3: Run tests 
All the tests should pass.
```commandline
python3 -m unittest discover -s tests 
```

### Step 4: Run with specific configurations
- An experiment configuration is an object of type
`ExperimentConfig` [source](src/experiment_config.py).

- All experiment configurations are located in the directory `experiment_config`, as 
`.py` files containing a `config` object that is an instance of `ExperimentConfig`.
You can use one of the existing configuration 
('bookcorpus', 'bookcorpus_and_wikipedia', ...) or
create your own (you can use an existing one as a reference).

To run the program with the experiment configuration of your choice,
run:
```commandline
python3 create_pmi_masking_vocab.py --experiment_config="your experiment config"
```
where you should replace "your experiment config" with the configuration 
file name without the `.py` suffix.


If the run is successful, the resulting vocabulary will be placed in `pmi_masking_vocabs` directory.

#### Logging
Running the program on large datasets might take a while.
Logging messages will be printed to console and to the file `log.log`.
Those logs can be used for measuring times after the run is complete,
or to track progress while the program is running.

