"""
    Name: NMFconv
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


import numpy as np
from copy import deepcopy
from tqdm import tnrange

from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.convModel import convModel
from NMFtoolbox.shiftOperator import shiftOperator
from NMFtoolbox.utils import EPS


def NMFconv(V, parameter=None):
    """
    Convolutive Non-Negative Matrix Factorization with Beta-Divergence and
    optional regularization parameters as described in chapter 3.7 of [2].
    The averaged activation updates are computed via the compact algorithm
    given in paragraph 3.7.3. For the sake of consistency, we use the notation
    from [3] instead of the one from the book.

    References
    ----------
    [2] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shun-ichi Amari
    "Nonnegative Matrix and Tensor Factorizations: Applications to
    Exploratory Multi-Way Data Analysis and Blind Source Separation"
    John Wiley and Sons, 2009.

    [3] Christian Dittmar and Meinard Müller "Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    V: array-like
        Matrix that shall be decomposed (typically a magnitude spectrogram
        of dimension numBins x numFrames)

    parameter: dict
        numComp           Number of NMFD components (denoted as R in [2])
        numIter           Number of NMFD iterations (denoted as L in [2])
        numTemplateFrames Number of time frames for the 2D-template (denoted as T in [2])
        initW             An initial estimate for the templates (denoted as W^(0) in [2])
        initH             An initial estimate for the gains (denoted as H^(0) in [2])
        beta              The beta parameter of the divergence:
                          -1 -> equals Itakura Saito divergence
                          0 -> equals Kullback Leiber divergence
                          1 -> equals Euclidean distance
        sparsityWeight    Strength of the activation sparsity
        uncorrWeight      Strength of the template uncorrelatedness

    Returns
    -------
    W: array-like
        List with the learned templates

    H: array-like
        Matrix with the learned activations

    cnmfY: array-like
        List with approximated component spectrograms

    costFunc: array-like
        The approximation quality per iteration

    """
    parameter = init_parameters(parameter)

    # use parameter nomenclature as in [2]
    K, M = V.shape

    T = parameter['numTemplateFrames']
    R = parameter['numComp']
    L = parameter['numIter']
    beta = parameter['beta']
    sparsityWeight = parameter['sparsityWeight']
    uncorrWeight = parameter['uncorrWeight']

    if 'numBins' not in parameter:
        parameter['numBins'] = K

    if 'numFrames' not in parameter:
        parameter['numFrames'] = M

    # use initial templates
    initW = initTemplates(parameter, 'random') if 'initW' not in parameter else parameter['initW']

    tensorW = np.zeros((initW[0].shape[0], R, T))

    # stack the templates into a tensor
    for r in range(R):
        tensorW[:, r, :] = deepcopy(initW[r])

    # use initial activations
    initH = initActivations(parameter, 'uniform') if 'initH' not in parameter else parameter['initH']

    # copy initial
    H = deepcopy(initH)

    # this is important to prevent initial jumps in the divergence measure
    V_tmp = V / (EPS + V.sum())

    costFunc = np.zeros(L)

    for iter in tnrange(L, desc='Processing'):
        # compute first approximation
        Lambda = convModel(tensorW, H)
        LambdaBeta = Lambda ** beta
        Q = V_tmp * LambdaBeta / Lambda

        costMat = V_tmp * (V_tmp ** beta - Lambda ** beta) / (EPS + beta) - (V_tmp ** (beta + 1) - Lambda ** (beta + 1)) / (EPS +
                                                                                                              beta + 1)
        costFunc[iter] = costMat.mean()

        for t in range(T):
            # respect zero index
            tau = t
            
            # use tau for shifting and t for indexing
            transpH = shiftOperator(H, tau).T

            numeratorUpdateW = Q @ transpH

            denominatorUpdateW = EPS + LambdaBeta @ transpH + uncorrWeight * \
                                 np.sum(tensorW[:, :, np.setdiff1d(np.arange(T), np.array([t]))], axis=2)

            tensorW[:, :, t] *= numeratorUpdateW / denominatorUpdateW

        numeratorUpdateH = convModel(np.transpose(tensorW, (1, 0, 2)), np.fliplr(Q))

        denominatorUpdateH = convModel(np.transpose(tensorW, (1, 0, 2)), np.fliplr(LambdaBeta)) + sparsityWeight + EPS
        H *= np.fliplr(numeratorUpdateH / denominatorUpdateH)

        # normalize templates to unit sum
        normVec = np.sum(np.sum(tensorW, axis=0), axis=1).reshape(-1, 1)

        tensorW = tensorW * 1 / (EPS + normVec)

    W = list()
    cnmfY = list()

    for r in range(R):
        W.append(tensorW[:, r, :])
        cnmfY.append(convModel(np.expand_dims(tensorW[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, cnmfY, costFunc


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function NMFconv for further information

    Returns
    -------
    parameter: dict
    """
    parameter = dict() if not parameter else parameter
    parameter['numIter'] = 30 if 'numIter' not in parameter else parameter['numIter']
    parameter['numComp'] = 3 if 'numComp' not in parameter else parameter['numComp']
    parameter['numTemplateFrames'] = 8 if 'numIter' not in parameter else parameter['numTemplateFrames']
    parameter['beta'] = 0 if 'beta' not in parameter else parameter['beta']
    parameter['sparsityWeight'] = 0 if 'sparsityWeight' not in parameter else parameter['sparsityWeight']
    parameter['uncorrWeight'] = 0 if 'uncorrWeight' not in parameter else parameter['uncorrWeight']

    return parameter
