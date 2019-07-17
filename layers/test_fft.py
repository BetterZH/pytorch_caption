import torch
torch.manual_seed(0)
# from _ext import th_fft
import pytorch_fft.fft as cfft
import numpy as np
import numpy.fft as nfft

def run_fft(x, z):
    if torch.cuda.is_available():
        y1, y2 = cfft.fft2(x, z)
        x_np = x.cpu().numpy().squeeze()
        y_np = nfft.fft2(x_np)
        assert np.allclose(y1.cpu().numpy(), y_np.real)
        assert np.allclose(y2.cpu().numpy(), y_np.imag)

        # assert np.allclose(y1[1,0].cpu().numpy(), nfft.fft2(x_np[1,0]).real)

        x0, z0 = cfft.ifft2(y1, y2)
        x0_np = nfft.ifft2(y_np)
        assert np.allclose(x0.cpu().numpy(), x0_np.real)
        assert np.allclose(z0.cpu().numpy(), x0_np.imag)

    else:
        print("Cuda not available, cannot test.")

def run_fft1(x, z):
    if torch.cuda.is_available():
        y1, y2 = cfft.fft(x, z)
        x_np = x.cpu().numpy().squeeze()
        y_np = nfft.fft(x_np)

        # print(y1.cpu().numpy())
        # print(y_np.real)

        assert np.allclose(y1.cpu().numpy(), y_np.real)
        assert np.allclose(y2.cpu().numpy(), y_np.imag)

        # assert np.allclose(y1[1,0].cpu().numpy(), nfft.fft2(x_np[1,0]).real)

        x0, z0 = cfft.ifft(y1, y2)
        x0_np = nfft.ifft(y_np)
        assert np.allclose(x0.cpu().numpy(), x0_np.real)
        assert np.allclose(z0.cpu().numpy(), x0_np.imag)

    else:
        print("Cuda not available, cannot test.")

def test_acc():
    batch = 3
    nch = 4
    n = 5
    m = 7
    x = torch.randn(batch*nch*n*m).view(batch, nch, n, m).cuda()
    z = torch.zeros(batch, nch, n, m).cuda()
    run_fft(x, z)
    run_fft(x.double(), z.double())

def test_acc1():
    batch = 3
    n = 5
    m = 7
    x = torch.randn(batch*n*m).view(batch, n, m).cuda()
    z = torch.zeros(batch, n, m).cuda()
    run_fft(x, z)
    run_fft(x.double(), z.double())

def test_acc2():
    batch = 3
    nch = 4
    n = 5
    x = torch.randn(batch*nch*n).view(batch, nch, n).cuda()
    z = torch.zeros(batch, nch, n).cuda()
    run_fft1(x, z)
    run_fft1(x.double(), z.double())

def test_acc3():
    batch = 3
    nch = 4
    n = 5

    x = torch.randn(batch * nch * n).view(batch, nch, n).cuda()
    y = torch.randn(batch * nch * n).view(batch, nch, n).cuda()

    x_i = torch.zeros(batch, nch, n).cuda()
    y_i = torch.zeros(batch, nch, n).cuda()

    if torch.cuda.is_available():

        x1_r, x1_i = cfft.fft(x, x_i)
        y1_r, y1_i = cfft.fft(y, y_i)

        print(x1_i)

        x0, z0 = cfft.ifft(x1_r*y1_r, x1_i*y1_i)

        print(z0)

    else:
        print("Cuda not available, cannot test.")

if __name__ == "__main__":
    test_acc3()