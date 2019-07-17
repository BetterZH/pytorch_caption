import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time

def test():
    img = mpimg.imread("/home/amds/Desktop/1.jpg")
    plt.imshow(img)
    plt.show()

# (n, height, width)
# (n, height, width, 3)
def vis_square(data, layer_num, channel):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    # plt.show()

    path = '/media/amds/data4/caption_result/test/conv_new/' + str(layer_num) + '_' + str(channel) + '.png'
    plt.savefig(path)

# numpy
# batch_size * channels * w *h
def vis_conv(conv_feat, layer_num):

    batch_size = conv_feat.shape[0]
    channels = conv_feat.shape[1]

    for i in range(batch_size):
        for j in range(channels):
            conv_feat_one = conv_feat[i,j]
            w = conv_feat_one.shape[0]
            h = conv_feat_one.shape[1]
            vis_square(conv_feat_one.reshape(1, w, h), layer_num, j)

def vis_one_channel(conv_feat):

    w = conv_feat.shape[0]
    h = conv_feat.shape[1]
    vis_square(conv_feat.reshape(1, w, h))


def show_img(img):
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test()
