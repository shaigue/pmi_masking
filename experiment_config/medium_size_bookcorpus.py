# TODO: save the parameters in order to extrapolate to the entire bookcorpus dataset, and also the
#   logs, so I can refine the scalability analysis
#   The idea is that I want to use the entire bookcorpus dataset as a test to the accuracy of my scalability analysis.
import config

# parameters of the pmi_masking algorithm, as used in the original paper.
max_ngram_size = 5
min_count_threshold = 10
vocab_size = 800_000
ngram_size_to_vocab_percent = {
    2: 50,
    3: 25,
    4: 12.5,
    5: 12.5,
}
# specific to this implementation
n_samples = 30_000_000
ngram_count_batch_size = 1_000_000
n_workers = 3
filter_ngram_count_threshold = 2
save_dir = config.DATA_DIR / 'medium_size_bookcorpus'
