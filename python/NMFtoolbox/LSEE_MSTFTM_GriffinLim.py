"""
    Name: LSEE_MSTFTMGriffinLim
    Date of Revision: June 2019
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

from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.inverseSTFT import inverseSTFT


def LSEE_MSTFTM_GriffinLim(X, parameter=None):
    """Performs one iteration of the phase reconstruction algorithm as
    described in [2].

    References
    ----------
    [2] Daniel W. Griffin and Jae S. Lim, Signal estimation
    from modified short-time fourier transform, IEEE
    Transactions on Acoustics, Speech and Signal Processing,
    vol. 32, no. 2, pp. 236-243, Apr 1984.

    The operation performs an iSTFT (LSEE-MSTFT) followed by STFT on the
    resynthesized signal.

    Parameters
    ----------
    X: array-like
        The STFT spectrogram to iterate upon

    parameter: dict
        blockSize:       The blocksize to use during analysis
        hopSize:         The used hopsize (denoted as S in [1])
        anaWinFunc:      The window used for analysis (denoted w in [1])
        synWinFunc:      The window used for synthesis (denoted w in [1])
        reconstMirror:   If this is enabled, we have to generate the
                         mirror spectrum by means of conjugation and flipping
        appendFrames:    If this is enabled, safety spaces have to be removed
                         after the iSTFT
        targetEnv:       If desired, we can define a time-signal mask from the
                         outside for better restoration of transients

    Returns
    -------
    Xout: array-like
        The spectrogram after iSTFT->STFT processing

    Pout: array-like
        The phase spectrogram after iSTFT->STFT processing

    res: array-like
        Reconstructed time-domain signal obtained via iSTFT
    """

    numBins, _ = X.shape
    parameter = init_parameters(parameter, numBins)

    Xout = deepcopy(X)
    A = abs(Xout)

    for k in range(parameter['numIterGriffinLim']):
        # perform inverse STFT
        res, _ = inverseSTFT(Xout, parameter)

        # perform forward STFT
        _, _, Pout = forwardSTFT(res.squeeze(), parameter)

        Xout = A * np.exp(1j * Pout)

    return Xout, Pout, res


def init_parameters(parameter, numBins):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function LSEE_MSTFTM_GriffinLim for further information

    Returns
    -------
    parameter: dict
    """
    if not parameter:
        parameter = dict()

    parameter['blockSize'] = 2048 if 'blockSize' not in parameter else parameter['blockSize']
    parameter['hopSize'] = 512 if 'hopSize' not in parameter else parameter['hopSize']
    parameter['winFunc'] = np.hanning(parameter['blockSize']) if 'winFunc' not in parameter else parameter['winFunc']

    # this controls if the upper part of the spectrum is given or should be
    # reconstructed by 'mirroring' (flip and conjugate) of the lower spectrum
    if 'reconstMirror' not in parameter:
        if numBins == parameter['blockSize']:
            parameter['reconstMirror'] = False
        elif numBins < parameter['blockSize']:
            parameter['reconstMirror'] = True

    parameter['appendFrames'] = True if 'appendFrames' not in parameter else parameter['appendFrames']
    parameter['analyticSig'] = False if 'analyticSig' not in parameter else parameter['analyticSig']
    parameter['numIterGriffinLim'] = 50 if 'numIterGriffinLim' not in parameter else parameter['numIterGriffinLim']

    return parameter