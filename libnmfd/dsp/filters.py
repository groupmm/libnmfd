import numpy as np
from typing import List, Tuple, Union

from libnmfd.utils import EPS


def alpha_wiener_filter(mixture_X: np.ndarray,
                        source_A: List[np.ndarray],
                        alpha: float = 1.2,
                        binarize=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Given a cell-array of spectrogram estimates as input, this function
    computes the alpha-related soft masks for extracting the sources. Details
    about this procedure are given in [1], further experimental studies in [2].

    References
    ----------
    [1] Antoine Liutkus and Roland Badeau:
    Generalized Wiener filtering with fractional power spectrograms, ICASPP 2015

    [2] Christian Dittmar, Jonathan Driedger, Meinard Müller, and Jouni Paulus
    An Experimental Approach to Generalized Wiener Filtering in Music Source Separation
    In Proceedings of the European Signal Processing Conference (EUSIPCO): 1743–1747, 2016.


    Parameters
    ----------
    mixture_X: array_like
        The mixture spectrogram (numBins x numFrames) (may be real-or complex-valued)

    source_A: list
        A list holding the equally sized spectrogram estimates of single sound sources (aka components)

    alpha: float
        The fractional power in rand [0 ... 2]

    binarize: bool
        If this is set to True, we binarize the masks

    Returns
    -------
    source_X: list
        A list of extracted source spectrograms

    softMasks: list
        A list with the extracted masks
    """

    num_bins, num_frames = mixture_X.shape
    num_comp = len(source_A)

    #  Initialize the mixture of the sources / components with a small constant
    mixtureA = EPS + np.zeros((num_bins, num_frames))

    softMasks = list()
    source_X = list()

    # Make superposition
    for k in range(num_comp):
        mixtureA += source_A[k] ** alpha

    # Compute soft masks and spectrogram estimates
    for k in range(num_comp):
        currSoftMask = (source_A[k] ** alpha) / mixtureA
        softMasks.append(currSoftMask.astype(np.float32))

        #  If desired, make this a binary mask
        if binarize:
            tmp = softMasks[k]
            softMasks[k] = tmp[tmp > (1.0/num_comp)] * 1

        #  And apply it to the mixture
        source_X.append(mixture_X * currSoftMask)

    return source_X, softMasks


def nema(A: np.ndarray,
         decay: Union[np.ndarray, float] = 0.9) -> np.ndarray:
    """This function takes a matrix of row-wise time series and applies a
    non-linear exponential moving average (NEMA) to each row. This filter
    introduces exponentially decaying slopes and is defined in eq. (3) from [2].

    The difference equation of that filter would be:
    y(n) = max( x(n), y(n-1)*(decay) + x(n)*(1-decay) )

    References
    ----------
    [1] Christian Dittmar, Patricio López-Serrano, and Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    A: np.ndarray
        The matrix with time series in its rows

    decay: np.ndarray or float
        The decay parameter in the range [0 ... 1], this can be given as a column-vector with individual decays per row
        or as a scalar

    Returns
    -------
    filtered: np.ndarray
        The result after application of the NEMA filter
    """
    # Prevent instable filter
    decay = max(0.0, min(0.9999999, decay))

    num_rows, num_cols = A.shape
    filtered = A.copy()

    for k in range(1, num_cols):
        store_row = filtered[:, k].copy()
        filtered[:, k] = decay * filtered[:, k - 1] + filtered[:, k] * (1 - decay)
        filtered[:, k] = np.maximum(filtered[:, k], store_row)

    return filtered


