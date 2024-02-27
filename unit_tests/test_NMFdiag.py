"""
    Name: test_NMFdiag
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
from NMFtoolbox.NMFdiag import NMFdiag
from NMFtoolbox.utils import EPS, make_monaural, pcmInt16ToFloat32Numpy, load_matlab_dict


def run_NMFdiag():
    inpPath = '../data/'
    matlabMatricesPath = 'matrices/NMFdiag/'

    filenameSource = 'Bees_Buzzing.wav'
    filenameTarget = 'Beatles_LetItBe.wav'

    # read signals
    fs, xs = wav.read(os.path.join(inpPath, filenameSource))
    fs, xt = wav.read(os.path.join(inpPath, filenameTarget))

    # make monaural if necessary
    xs = make_monaural(xs)
    xt = make_monaural(xt)

    # convert wavs from int16 to float32
    xs = pcmInt16ToFloat32Numpy(xs)
    xt = pcmInt16ToFloat32Numpy(xt)

    paramSTFT = dict()
    paramSTFT['blockSize'] = 2048
    paramSTFT['hopSize'] = 1024
    paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
    paramSTFT['reconstMirror'] = True
    paramSTFT['appendFrame'] = True
    paramSTFT['numSamples'] = len(xt)

    # STFT computation
    Xs, As, Ps = forwardSTFT(xs, paramSTFT)
    Xt, At, Pt = forwardSTFT(xt, paramSTFT)

    # get dimensions and time and freq resolutions
    _, numTargetFrames = Xt.shape

    # initialize activations randomly
    # load randomly initialized matrix on MATLAB
    H0 = load_matlab_dict(os.path.join(matlabMatricesPath, 'H0.mat'), 'H0')

    # init templates by source frames
    W0 = As * 1./ (EPS + np.sum(As, axis=0))

    paramNMFdiag = dict()
    paramNMFdiag['fixW'] = True
    paramNMFdiag['numOfIter'] = 3
    paramNMFdiag['continuity'] = dict()
    paramNMFdiag['continuity']['polyphony'] = 10
    paramNMFdiag['continuity']['length'] = 7
    paramNMFdiag['continuity']['grid'] = 1
    paramNMFdiag['continuity']['sparsen'] = [1, 7]

    # call the reference implementation as provided by Jonathan Driedger
    # with divergence update rules
    nmfdiagW_div, nmfdiagH_div = NMFdiag(At, W0, H0, paramNMFdiag)

    python_res = {'nmfdiagW_div': nmfdiagW_div,
                  'nmfdiagH_div': nmfdiagH_div,
                  }

    return python_res