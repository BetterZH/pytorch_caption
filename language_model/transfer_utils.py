import numpy as np

transer_matrix = None

def get_transer_score(seq):

    global transer_matrix

    if transer_matrix is None:
        path = '/home/public/dataset/aic/trans_prob.npz'
        transer_matrix = np.load(path)['x']

    # batch_size * beam_size * seq_length
    N, B, D = seq.shape
    out_beam_sum_scores = []
    out_beam_mean_scores = []

    for i in range(N):

        beam_sum_scores = []
        beam_mean_scores = []

        for k in range(B):
            score = 0
            len = 0

            for j in range(D-1):

                ix1 = seq[i, k, j]
                ix2 = seq[i, k, j+1]

                score += np.log(transer_matrix[ix1, ix2])
                len += 1
                if ix2 == 0:
                    break

            if len == 0:
                len = 1

            beam_sum_scores.append(score)
            beam_mean_scores.append(score/len)

        out_beam_sum_scores.append(beam_sum_scores)
        out_beam_mean_scores.append(beam_mean_scores)

    # out_beam_sum_scores : batch_size * beam_size
    # out_beam_mean_scores : batch_size * beam_size
    return out_beam_sum_scores, out_beam_mean_scores




