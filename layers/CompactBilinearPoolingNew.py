from __future__ import absolute_import, division, print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn.init as init
import math
import pytorch_fft.fft.autograd as fft
import numpy as np
from ipdb import set_trace as st

# https://github.com/jnhwkim/cbp/blob/master/CompactBilinearPooling.lua
# https://github.com/locuslab/pytorch_fft
# tf code: https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling/blob/master/compact_bilinear_pooling.py
# pytorch sparse matrix: http://pytorch.org/docs/master/sparse.html?highlight=mm#torch.sparse.FloatTensor.spmm
# Primary reference : tensorflow CCP https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling

"""
Compute compact bilinear pooling over two bottom inputs. Reference:
Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
Conference on Computer Vision and Pattern Recognition (2016).
Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).
Mainly
"""

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a (sparse) matrix used for tensor Sketch Count operation in compact bilinear pooling,
     pytorch dont support autograd of sparse tensor, wo we use dense tensor instead.
    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.

    Returns:
        torch Variable, a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    assert (rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s))
    assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    i = torch.LongTensor(indices).t()
    v = torch.IntTensor(rand_s)
    sparse_sketch_matrix = torch.sparse.IntTensor(i, v, torch.Size([input_dim, output_dim] )).to_dense().float().cuda()
    # I used to want to used sparse matrix, but the autograd is not suported.
    return Variable(sparse_sketch_matrix)


class CompactBilinearPooling(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True):
        """
        Args:
            output_dim1: output dimension for compact bilinear pooling.
            output_dim2: output dimension for compact bilinear pooling.
            sum_pool: (Optional) If True, sum the output along height and width
                      dimensions and return output shape [batch_size, output_dim].
        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.
    """

        super(CompactBilinearPooling, self).__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        # Step 0: Generate vectors and sketch matrix for tensor count sketch
        # This is only done once during graph construction, and fixed during each
        # operation
        np.random.seed(1)
        self.rand_h_1 = np.random.randint(output_dim, size=input_dim1)
        np.random.seed(3)
        self.rand_s_1 = 2 * np.random.randint(2, size=input_dim1) - 1
        self.sparse_sketch_matrix1 = _generate_sketch_matrix(self.rand_h_1, self.rand_s_1, self.output_dim)
        np.random.seed(5)
        self.rand_h_2 = np.random.randint(output_dim, size=input_dim2)
        np.random.seed(7)
        self.rand_s_2 = 2 * np.random.randint(2, size=input_dim2) - 1
        self.sparse_sketch_matrix2 = _generate_sketch_matrix(self.rand_h_2, self.rand_s_2, self.output_dim)

        self.f1 = fft.Fft()
        self.f2 = fft.Fft()

        self.invf = fft.Ifft()


    def forward(self, x, y):  # bottom1, bottom2):
        """
        Compute compact bilinear pooling over two bottom inputs.
        Args:
            bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width ] NCHW. (tensorflow is NHWC)
            bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width ] NCHW.
        Returns:
            Compact bilinear pooled results of shape [batch_size, output_dim] or
            [batch_size, height, width, output_dim], depending on `sum_pool`.
        """
        # self.x = x  # just to test x.gradient
        # self.y = y
        # bottom1 = x.clone()
        # bottom2 = y.clone()
        bottom1 = x
        bottom2 = y
        size1 = list(bottom1.data.size())
        size2 = list(bottom2.data.size())
        print('size1', size1)
        print('size2', size2)
        assert size1[1] == self.input_dim1
        assert size2[1] == self.input_dim2

        # mv channel axis to last axis, in order for reshape to [-1, input_dim],
        # out shape: [batch_size,  height, width, input_dim]
        bottom1 = x.permute(0, 2, 3, 1).contiguous()
        bottom2 = y.permute(0, 2, 3, 1).contiguous()
        permuted_size = list(bottom1.data.size())
        # Step 1: Flatten the input tensors, out shape: [-1, input_dim]
        bottom1_flat = bottom1.view(size1[0] * size1[2] * size1[3], self.input_dim1)
        bottom2_flat = bottom2.view(size2[0] * size2[2] * size2[3], self.input_dim2)

        # compute  Count Sketch
        print(bottom1_flat.data.size())
        print(self.sparse_sketch_matrix1.data.size())
        sketch1 = bottom1_flat.mm(self.sparse_sketch_matrix1)  # (NxInput_dim) x (Input_dim x Output_dim)
        sketch2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        # Step 2: 1-dim FFT on the input_dim aixs
        fft1_real, fft1_image = self.f1(sketch1, Variable(torch.zeros(sketch1.size())).cuda())
        fft2_real, fft2_image = self.f2(sketch2, Variable(torch.zeros(sketch2.size())).cuda())
        # Step 3: Elementwise product between complex numbers, (a+bi)(c+di)=(ac-bd)+(bc+ad)i
        fft_real = fft1_real * fft2_real - fft1_image * fft2_image
        fft_image = fft2_real * fft1_image + fft1_real * fft2_image
        # Step 4: Inverse FFT and reshape back
        cbp_real, cbp_image = self.invf(fft_real, fft_image)
        # reshape: [batch_size,  height, width, output_dim]
        output_shape = list(np.multiply(permuted_size, [1, 1, 1, 0]) + [0, 0, 0, self.output_dim])
        print('output_shape', output_shape)
        print(cbp_real.size())
        # and what to do with the image part?abundon? will it interupt the gradient flow?
        cbp = cbp_real + cbp_image * 0
        print('cbp size', cbp.size())
        cbp = cbp.view(output_shape)
        # print(cbp.size())
        # permute back: [batch_size, output_dim, height, width]
        cbp = cbp.permute(0, 3, 1, 2).contiguous()
        # print(cbp.size())
        # test backward  gradient
        # z = cbp.sum()
        # z = (fft_real + fft_image).sum()
        # z.backward(retain_graph=True)
        # print(x.grad)
        # print(y.grad)

        # Step 5: Sum pool over spatial dimensions, if specified
        # out shape: [batch_size, output_dim]
        print('cbp_size', cbp.size())
        if self.sum_pool:
            cbp = cbp.sum(2).sum(2)
        print('cbp_size', cbp.size())

        return cbp
