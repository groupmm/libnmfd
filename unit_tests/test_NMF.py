"""
    Name: test_NMF
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
from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.NMF import NMF
from NMFtoolbox.NMFconv import NMFconv
from NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy


def run_NMF():
    inpPath = '../data/'
    filename = 'runningExample_AmenBreak.wav'

    # read signals
    fs, x = wav.read(os.path.join(inpPath, filename))

    # make monaural if necessary
    x = make_monaural(x)

    # convert wavs from int16 to float32
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
    X, A, P = forwardSTFT(x, paramSTFT)

    # get dimensions and time and freq resolutions
    numBins, numFrames = X.shape
    deltaT = paramSTFT['hopSize'] / fs
    deltaF = fs / paramSTFT['blockSize']

    # Apply NMF variants to STFT magnitude
    # set common parameters
    numComp = 3
    numIter = 3
    numTemplateFrames = 8

    # generate initial guess for templates
    paramTemplates = dict()
    paramTemplates['deltaF'] = deltaF
    paramTemplates['numComp'] = numComp
    paramTemplates['numBins'] = numBins
    paramTemplates['numTemplateFrames'] = numTemplateFrames
    initW = initTemplates(paramTemplates, 'drums')

    # generate initial activations
    paramActivations = dict()
    paramActivations['numComp'] = numComp
    paramActivations['numFrames'] = numFrames

    initH = initActivations(paramActivations, 'uniform')

    # NMFconv parameters
    paramNMFconv = dict()
    paramNMFconv['numComp'] = numComp
    paramNMFconv['numFrames'] = numFrames
    paramNMFconv['numIter'] = numIter
    paramNMFconv['numTemplateFrames'] = numTemplateFrames
    paramNMFconv['initW'] = initW
    paramNMFconv['initH'] = initH
    paramNMFconv['beta'] = 0

    # NMFconv core method
    nmfconvW, _, nmfconvV, _ = NMFconv(A, paramNMFconv)

    # alpha-Wiener filtering
    nmfconvA, _ = alphaWienerFilter(A, nmfconvV, 1)

    W0 = np.concatenate(nmfconvW, axis=1)

    # set common parameters
    numComp = W0.shape[1]
    numIter = 3

    # generate random initialization for activations
    paramActivations = dict()
    paramActivations['numComp'] = numComp
    paramActivations['numFrames'] = numFrames
    initH = initActivations(paramActivations, 'uniform')

    # store common parameters
    paramNMF = dict()
    paramNMF['numComp'] = numComp
    paramNMF['numFrames'] = numFrames
    paramNMF['numIter'] = numIter
    paramNMF['initW'] = W0
    paramNMF['initH'] = initH

    # NMF with Euclidean Distance cost function
    paramNMF['costFunc'] = 'EucDist'
    nmfEucDistW, nmfEucDistH, nmfEucDistV = NMF(A, paramNMF)

    # NMF with KLDiv Distance cost function
    paramNMF['costFunc'] = 'KLDiv'
    nmfKLDivW, nmfKLDivH, nmfKLDivV = NMF(A, paramNMF)

    # NMF with ISDiv Distance cost function
    paramNMF['costFunc'] = 'ISDiv'
    nmfISDivW, nmfISDivH, nmfISDivV = NMF(A, paramNMF)

    python_res = {'nmfEucDistW': nmfEucDistW,
                  'nmfEucDistH': nmfEucDistH,
                  'nmfEucDistV': nmfEucDistV,
                  'nmfKLDivW': nmfKLDivW,
                  'nmfKLDivH': nmfKLDivH,
                  'nmfKLDivV': nmfKLDivV,
                  'nmfISDivW': nmfISDivW,
                  'nmfISDivH': nmfISDivH,
                  'nmfISDivV': nmfISDivV}

    return python_res
