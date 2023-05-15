
# what to we want to estimate?
# there are several steps.
# time:
# - total time
# - time for each step. this will help us understand where our estimtates where wrong.

# space:
# - the space for all the chunck after the first stage.
# - how many ngrams are in each chunk, how many are removed in prunning
# - the space after the compute pmi_score. maybe to save. maybe we can even devide it by ngram_size.
# - understand the how the number of ngrams increases as the number of tokens increases.
# - how many ngrams of each size there are in the table after merging.
# we want to do the estimates using the total number of tokens in the dataset.
# TODO: how do we get the vocabulary size?
# TODO: I don't take into considiration the filtering I do of ngrams of size 1 or less. this significantly effects that computation.
# TODO: for saving memory & time in steps after the merging of all of the tables, I can perform the filter of ngrams of
#    by their size before starting the computations. -- this is a quick fix.
# TODO: At the start of running the pipeline, log the configuration name. This will help us analyze and split the logs.
# TODO: create a module named `logs_processing` that will contain functions to extract required information from the logs.
# actual = ?  # TODO: how to get the actual number of tokens and  from the logs? and the number of ngrams?
# TODO: make sure that all the numbers required for the processing will appear in the logs.
# next step is to extract the experiment data. what data do we need?
# - the total number of tokens processed (this could be logged in the counting ngrams in chunks phase)
# - time, per step and ngram size the procedure took.
# -

# TODO: a lot of the code here should go to the process_logs.py and this should contain more of the estimation procedure
def estimate_space():
    # TODO:
    pass


def actual_space():
    # TODO: extract the actual space used by an experiment. this is the maximum between the total size of the files
    #   for the batch counts and the final size of the table after computing pmi scores.
    pass
