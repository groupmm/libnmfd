"""
    Name: test_STFT
    Date of Revision: Jul 2019
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

import os
import numpy as np
import scipy.io.wavfile as wav

from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.inverseSTFT import inverseSTFT
from NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy


def run_STFT():
    inpPath = '../data/'
    filename = 'runningExample_IGotYouMixture.wav'

    # read signals
    fs, x = wav.read(os.path.join(inpPath, filename))

    # make monaural if necessary
    x = make_monaural(x)

    # convert wavs from int16 to float32
    x = pcmInt16ToFloat32Numpy(x)

    paramSTFT = dict()
    paramSTFT['blockSize'] = 2048
    paramSTFT['hopSize'] = 1024
    paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
    paramSTFT['reconstMirror'] = True
    paramSTFT['appendFrame'] = True
    paramSTFT['numSamples'] = len(x)

    # STFT computation
    X, A, P = forwardSTFT(x, paramSTFT)

    # inverse STFT
    y, _ = inverseSTFT(X, paramSTFT)

    python_res = {'X': X, 'A': A, 'P': P, 'y': y}

    return python_res


