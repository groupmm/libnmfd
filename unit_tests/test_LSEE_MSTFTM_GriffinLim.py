"""
    Name: test_initTemplates
    Date of Revision: Jul 2019
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


import os
import numpy as np
import scipy.io.wavfile as wav

from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy
from NMFtoolbox.LSEE_MSTFTM_GriffinLim import LSEE_MSTFTM_GriffinLim


def run_LSEE_MSTFTM_GriffinLim():
    inpPath = '../data/'
    filename = 'runningExample_IGotYouMixture.wav'

    # read signal
    fs, x = wav.read(os.path.join(inpPath, filename))

    # make monaural if necessary
    x = make_monaural(x)

    # convert wav from int16 to float32
    x = pcmInt16ToFloat32Numpy(x)


    # spectral parameters
    paramSTFT = dict()
    paramSTFT['blockSize'] = 2048
    paramSTFT['hopSize'] = 512
    paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
    paramSTFT['reconstMirror'] = True
    paramSTFT['appendFrame'] = True
    paramSTFT['numSamples'] = len(x)

    # STFT computation
    _, A, _ = forwardSTFT(x, paramSTFT)

    Xout, Pout, res = LSEE_MSTFTM_GriffinLim(A, paramSTFT)

    python_res = dict()
    python_res['res'] = res

    return python_res
