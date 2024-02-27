"""
    Name: HPSS_KAM
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

from copy import deepcopy
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import convolve2d

from NMFtoolbox.alphaWienerFilter import alphaWienerFilter


def HPSS_KAM_Fitzgerald(X, numIter=1, kernDim=17, useMedian=False, alphaParam=1.0):
    """This re-implements the KAM-based HPSS-algorithm described in [2]. This is
    a generalization of the median-filter based algorithm first presented in [3].
    Our own variant of this algorithm [4] is also supported.

    References
    ----------
    [2] Derry FitzGerald, Antoine Liutkus, Zafar Rafii, Bryan Pardo,
    and Laurent Daudet, "Harmonic/percussive separation
    using Kernel Additive Modelling", in Irish Signals
    and Systems Conference (IET), Limerick, Ireland, 2014, pp. 35�40.

    [3] Derry FitzGerald, "Harmonic/percussive separation using
    median filtering," in Proceedings of the International
    Conference on Digital Audio Effects (DAFx),
    Graz, Austria, 2010, pp. 246-253.

    [4] Christian Dittmar, Jonathan Driedger, Meinard Müller,
    and Jouni Paulus, "An experimental approach to generalized
    wiener filtering in music source separation," EUSIPCO 2016.

    Parameters
    ----------
    X: array-like
        Input mixture magnitude spectrogram

    numIter: int
        The number of iterations

    kernDim: int
        The kernel dimensions

    useMedian: bool
        If True, reverts to FitzGerald's old method

    :param alphaParam: float
        The alpha-Wiener filter exponent


    Returns
    -------
    kamX: array-like
        List containing the percussive and harmonic estimate

    Kern: array-like
        The kernels used for enhancing percussive and harmonic part

    KernOrd: int
        The order of the kernels
    """

    # prepare data for the KAM iterations
    kamX = list()
    KernOrd = np.ceil(kernDim / 2).astype(np.int)

    # construct median filter kernel
    Kern = np.full((kernDim, kernDim), False, dtype=bool)
    Kern[KernOrd - 1, :] = True

    # construct low-pass filter kernel
    K = np.hanning(kernDim)
    K /= K.sum()

    # initialize first version with copy of original
    kamX.append(deepcopy(X))
    kamX.append(deepcopy(X))

    for iter in range(numIter):

        if useMedian:
            # update estimates via method from [1]
            kamX[0] = median_filter(kamX[0], footprint=Kern.T, mode='constant')
            kamX[1] = median_filter(kamX[1], footprint=Kern, mode='constant')

        else:
            # update estimates via method from [2]
            kamX[0] = convolve2d(kamX[0], K.reshape(-1, 1), mode='same')
            kamX[1] = convolve2d(kamX[1], K.reshape(1, -1), mode='same')

        # apply alpha Wiener filtering
        kamX, _ = alphaWienerFilter(X, kamX, alphaParam)

    return kamX, Kern, KernOrd
