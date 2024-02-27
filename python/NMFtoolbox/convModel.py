"""
    Name: convModel
    Date: June 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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

import numpy as np

from NMFtoolbox.shiftOperator import shiftOperator
from NMFtoolbox.utils import EPS


def convModel(W, H):
    """Convolutive NMF model implementing the eq. (4) from [2]. Note that it can
    also be used to compute the standard NMF model in case the number of time
    frames of the templates equals one.

    References
    ----------
    [2] Christian Dittmar and Meinard Müller "Reverse Engineering the Amen
    Break " Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 15311543, 2016.


    Parameters
    ----------
    W: array-like
        Tensor holding the spectral templates which can be interpreted as a set of
        spectrogram snippets with dimensions: numBins x numComp x numTemplateFrames

    H: array-like
        Corresponding activations with dimensions: numComponents x numTargetFrames

    Returns
    -------
    lamb: array-like
        Approximated spectrogram matrix

    """
    # the more explicit matrix multiplication will be used
    numBins, numComp, numTemplateFrames = W.shape
    numComp, numFrames = H.shape

    # initialize with zeros
    lamb = np.zeros((numBins, numFrames))

    # this is doing the math as described in [2], eq (4)
    # the alternative conv2() method does not show speed advantages

    for k in range(numTemplateFrames):
        multResult = W[:, :, k] @ shiftOperator(H, k)
        lamb += multResult

    lamb += EPS

    return lamb
