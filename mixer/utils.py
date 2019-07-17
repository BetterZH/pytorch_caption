import math

# util functions useful to compute BLEU/ROUGE metrics (cpu-based).
#
# the first n entry (integers) of the input vector vec
# are hashed to a unique integer;
# e.g. n=3, (i, j, k) -> (i-1)*V^2 + (j-1)*V + (k - 1) + 1
# V is the vocabulary size
def compute_hash(vec, n, V):
    hash = 0
    for cnt in xrange(n):
        hash = hash + (vec[cnt] - 1) * math.pow(V, n- cnt)
    return hash

# compute ngram counts
# input is a 1D tensor storing the indexes of the words int the sequence.
# if skip id is not nil, then the ngram is skipped.
def get_counts(input, nn, V, skip_id, output):
    sequence_length = input.size(0)
    out = output
    for tt in xrange(sequence_length - nn + 1):
        curr_window = input[tt: tt + nn]
        if skip_id is None or curr_window.eq(skip_id).sum() == 0:
            hash = compute_hash(curr_window, nn, V)
        if out[hash] is None:
            out[hash] = 0
        else:
            out[hash] = out[hash] + 1

    return out

