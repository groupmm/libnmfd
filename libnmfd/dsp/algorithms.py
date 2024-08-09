import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from tqdm import tnrange
from typing import Tuple, List

from libnmfd.dsp.filters import alpha_wiener_filter
from libnmfd.dsp.transforms import forward_stft, inverse_stft


def griffin_lim(X: np.ndarray,
                num_iter: int = 50,
                block_size: int = 2048,
                hop_size: int = 512,
                win: np.ndarray = None,
                append_frames: bool = True,
                analytic_sig: bool = False,
                **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs one iteration of the phase reconstruction algorithm as
    described in [2].

    References
    ----------
    [1] Daniel W. Griffin and Jae S. Lim
    Signal estimation from modified short-time fourier transform
    IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243, Apr 1984.

    The operation performs an iSTFT (LSEE-MSTFT) followed by STFT on the resynthesized signal.

    Parameters
    ----------
    X: np.ndarray
        The STFT spectrogram to iterate upon

    num_iter: int
        Number of iterations

    block_size: int
        The block size to use during analysis

    hop_size: int
        The used hop size (denoted as S in [1])

    win: np.ndarray
        Window function

    append_frames: bool
        If this is enabled, safety spaces have to be removed after the iSTFT

    analytic_sig: bool
        If this is set to True, we want the analytic signal

    Returns
    -------
    Xout: np.ndarray
        The spectrogram after iSTFT->STFT processing

    Pout: np.ndarray
        The phase spectrogram after iSTFT->STFT processing

    res: np.ndarray
        Reconstructed time-domain signal obtained via iSTFT        
    """
    num_bins, _ = X.shape
    win = np.hanning(block_size) if win is None else win

    # this controls if the upper part of the spectrum is given or should be
    # reconstructed by 'mirroring' (flip and conjugate) of the lower spectrum
    reconst_mirror = False if num_bins == block_size else True

    Xout = X.copy()
    A = np.abs(Xout)

    res = None
    Pout = None

    for k in range(num_iter):
        # perform inverse STFT
        res, _ = inverse_stft(X=Xout,
                              block_size=block_size,
                              hop_size=hop_size,
                              ana_win_func=win,
                              syn_win_func=win,
                              reconst_mirror=reconst_mirror,
                              append_frames=append_frames,
                              analytic_sig=analytic_sig,
                              **kwargs)

        # perform forward STFT
        _, _, Pout = forward_stft(x=res.squeeze(),
                                  block_size=block_size,
                                  hop_size=hop_size,
                                  win=win,
                                  reconst_mirror=reconst_mirror,
                                  append_frames=append_frames)

        Xout = A * np.exp(1j * Pout)

    return Xout, Pout, res

def hpss_kam_fitzgerald(X: np.ndarray,
                        num_iter: int = 1,
                        kern_dim: int = 17,
                        use_median: bool = False,
                        alpha_param: float = 1.0) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """This re-implements the KAM-based HPSS-algorithm described in [1]. This is
    a generalization of the median-filter based algorithm first presented in [2].
    Our own variant of this algorithm [3] is also supported.

    References
    ----------
    [1] Derry FitzGerald, Antoine Liutkus, Zafar Rafii, Bryan Pardo, and Laurent Daudet
    Harmonic/Percussive Separation using Kernel Additive Modelling
    Irish Signals and Systems Conference (IET), Limerick, Ireland, 2014, pp. 35�40.

    [2] Derry FitzGerald
    Harmonic/Percussive Separation using Median Filtering
    In Proceedings of the International Conference on Digital Audio Effects (DAFx), Graz, Austria, 2010, pp. 246-253.

    [3] Christian Dittmar, Jonathan Driedger, Meinard Müller, and Jouni Paulus
    An Experimental Approach to Generalized Wiener Filtering in Music Source Separation
    In Proceedings of the European Signal Processing Conference (EUSIPCO): 1743–1747, 2016.

    Parameters
    ----------
    X: np.ndarray
        Input mixture magnitude spectrogram

    num_iter: int
        The number of iterations

    kern_dim: int
        The kernel dimensions

    use_median: bool
        If True, reverts to FitzGerald's old method

    alpha_param: float
        The alpha-Wiener filter exponent

    Returns
    -------
    kam_X: list
        List containing the percussive and harmonic estimate

    kern: np.ndarray
        The kernels used for enhancing percussive and harmonic part

    kern_ord: int
        The order of the kernels
    """

    # prepare data for the KAM iterations
    kam_X = list()
    kern_ord = np.ceil(kern_dim / 2).astype(np.int32)

    # construct median filter kernel
    kern = np.full((kern_dim, kern_dim), False, dtype=bool)
    kern[kern_ord - 1, :] = True

    # construct low-pass filter kernel
    K = np.hanning(kern_dim)
    K /= K.sum()

    # initialize first version with copy of original
    kam_X.append(X.copy())
    kam_X.append(X.copy())

    for _ in tnrange(num_iter, desc='Processing'):
        if use_median:
            # update estimates via method from [1]
            kam_X[0] = median_filter(kam_X[0], footprint=kern.T, mode='constant')
            kam_X[1] = median_filter(kam_X[1], footprint=kern, mode='constant')

        else:
            # update estimates via method from [2]
            kam_X[0] = convolve2d(kam_X[0], K.reshape(-1, 1), mode='same')
            kam_X[1] = convolve2d(kam_X[1], K.reshape(1, -1), mode='same')

        # apply alpha Wiener filtering
        kam_X, _ = alpha_wiener_filter(X, kam_X, alpha_param)

    return kam_X, kern, kern_ord

