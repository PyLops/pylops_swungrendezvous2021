import time
import numpy as np
import cupy as cp
from pylops.utils.wavelets import ricker
from pylops.signalprocessing import Convolve1D
from pylops.optimization.sparsity import FISTA


def deconv(nt):
    print('Deconvolution with nt=%d....' % nt)
    # Input parameters
    t_min, t_max = -4.0, 4.0 # time axis
    f_central = 15 # central frequency of wavelet
    t_delay = 1 # 
    t_delay_wav = np.array([0.0, 0.6, 1.2])[:, np.newaxis] # times of spikes

    # Time axis
    t = np.linspace(t_min, t_max, nt)

    # Model vector
    sigma = 1 / (np.pi * f_central) ** 2
    wav = np.exp(-((t - t_delay_wav) ** 2) / sigma)
    model = wav[0] - 1.5*wav[1] + wav[2]

    # Impulse response (Ricker wavelet)
    impulse_response_full = ricker(t[nt//2:], np.sqrt(1 / (np.pi**2 * sigma)))[0]
    impulse_response = impulse_response_full[np.argmax(impulse_response_full)-100:
                                             np.argmax(impulse_response_full)+101]

    # Convolution operator
    Rop = Convolve1D(nt, impulse_response, 
                    offset=np.argmax(impulse_response),
                    method='fft')
    data = Rop * model

    # Model reconstruction - Inverse problem with numpy
    tstart_np = time.time()
    model_reconstructed = FISTA(Op=Rop,
                                data=data,
                                eps=1e-1,
                                niter=50, 
                                show=False)[0]
    telapsed_np = time.time() - tstart_np

    # Model reconstruction - Inverse problem with cupy
    Rop_cp = Convolve1D(nt, cp.asarray(impulse_response), 
                      offset=np.argmax(impulse_response),
                      method='fft')
    data_cp = cp.asarray(data)

    tstart_cp = time.time()
    model_reconstructed_cp = FISTA(Op=Rop_cp,
                                   data=data_cp,
                                   eps=1e-1,
                                   niter=50, 
                                   show=False)[0]
    telapsed_cp = time.time() - tstart_cp

    return telapsed_np, telapsed_cp
