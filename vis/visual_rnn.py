import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def test1():
    x = np.arange(-np.pi,np.pi,0.01)
    y = np.sin(x)
    plt.plot(x,y,'g')
    plt.grid(True)
    plt.show()

# data : vocab_size
def show_probs(data):
    x_shape = data.shape[0]
    x = np.linspace(0, x_shape-1, x_shape)
    plt.bar(x, data)
    plt.grid(True)
    plt.ylim(data.min(), data.max())
    plt.show()

def cut0(caption):
    tokens = []
    for i in range(len(caption)):
        tokens.append(caption[i])
    return tokens

# list_logprob : (seq_length + 2) * ensemble_size * beam_size * (vocab_size + 1)
# seq : seq_length + 2
# sents : seq_length + 2
def save_data_fig(filepath, list_logprob, seq, sent):

    tokens = cut0(sent)
    seq_len = seq.size(0)
    plt.subplot(seq_len, 5, 1)
    x_shape = list_logprob.shape[0]
    x = np.linspace(0, x_shape - 1, x_shape)
    plt.bar(x, list_logprob)
    plt.grid(True)
    plt.ylim(list_logprob.min(), list_logprob.max())
    plt.savefig(filepath)
    # plt.show()

# batch_size * vocab_size
def show_all_probs(data):
    for i in range(data.shape[0]):
        show_probs(data[i])

# batch_size * vocab_size
def show_all_probs1(data):
    for i in range(data.shape[0]):
        show_probs(data[i])


if __name__ == '__main__':
    pass