"""
    Name: NMFD
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

from NMFtoolbox.convModel import convModel
from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.shiftOperator import shiftOperator
from NMFtoolbox.utils import EPS


def NMFD(V, parameter=None, paramConstr=None):
    """Non-Negative Matrix Factor Deconvolution with Kullback-Leibler-Divergence
    and fixable components. The core algorithm was proposed in [2], the
    specific adaptions are used in [3].

    References
    ----------
    [2] Paris Smaragdis "Non-negative Matix Factor Deconvolution;
    Extraction of Multiple Sound Sources from Monophonic Inputs".
    International Congress on Independent Component Analysis and Blind Signal
    Separation (ICA), 2004

    [3] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    V: array-like
        Matrix that shall be decomposed (typically a magnitude spectrogram of dimension
        numBins x numFrames)

    parameter: dict
        numComp            Number of NMFD components (denoted as R in [3])
        numIter            Number of NMFD iterations (denoted as L in [3])
        numTemplateFrames  Number of time frames for the 2D-template (denoted as T in [3])
        initW              An initial estimate for the templates (denoted as W^(0) in [3])
        initH              An initial estimate for the gains (denoted as H^(0) in [3])

    paramConstr: dict
        If this is given, it should contain parameters for constraints

    Returns
    -------
    W: array-like
        List with the learned templates

    H: array-like
        Matrix with the learned activations

    nmfdV: array-like
        List with approximated component spectrograms

    costFunc: array-like
        The approximation quality per iteration

    tensorW: array-like
        If desired, we can also return the tensor
    """
    # use parameter nomenclature as in [2]
    K, M = V.shape
    T = parameter['numTemplateFrames']
    R = parameter['numComp']
    L = parameter['numIter']

    parameter, paramConstr = init_parameters(parameter, paramConstr)

    initW = parameter['initW']
    initH = parameter['initH']

    tensorW = np.zeros((initW[0].shape[0], R, T))

    costFunc = np.zeros(L)

    # stack the templates into a tensor
    for r in range(R):
        tensorW[:, r, :] = initW[r]

    # the activations are matrix shaped
    H = deepcopy(initH)

    # create helper matrix of all ones (denoted as J in eq (5,6) in [2])
    onesMatrix = np.ones((K, M))

    # this is important to prevent initial jumps in the divergence measure
    V_tmp = V / (EPS + V.sum())

    for iter in tnrange(L, desc='Processing'):
        # if given from the outside, apply soft constraints
        if 'funcPointerPreProcess' in paramConstr:
            tensorW, H = paramConstr['funcPointerPreProcess'](tensorW, H, iter, L, paramConstr)

        # compute first approximation
        Lambda = convModel(tensorW, H)

        # store the divergence with respect to the target spectrogram
        costMat = V_tmp * np.log(1.0 + V_tmp/(Lambda+EPS)) - V_tmp + Lambda
        costFunc[iter] = costMat.mean()

        # compute the ratio of the input to the model
        Q = V_tmp / (Lambda + EPS)

        # accumulate activation updates here
        multH = np.zeros((R, M))

        # go through all template frames
        for t in range(T):
            # use tau for shifting and t for indexing
            tau = deepcopy(t)

            # The update rule for W as given in eq. (5) in [2]
            # pre-compute intermediate, shifted and transposed activation matrix
            transpH = shiftOperator(H, tau).T

            # multiplicative update for W
            multW = Q @ transpH / (onesMatrix @ transpH + EPS)

            if not parameter['fixW']:
                tensorW[:, :, t] *= multW

            # The update rule for W as given in eq. (6) in [2]
            # pre-compute intermediate matrix for basis functions W
            transpW = tensorW[:, :, t].T

            # compute update term for this tau
            addW = (transpW @ shiftOperator(Q, -tau)) / (transpW @ onesMatrix + EPS)

            # accumulate update term
            multH += addW

        # multiplicative update for H, with the average over all T template frames
        if not parameter['fixH']:
            H *= multH / T

        # if given from the outside, apply soft constraints
        if 'funcPointerPostProcess' in paramConstr:
            tensorW, H = paramConstr['funcPointerPostProcess'](tensorW, H, iter, L, paramConstr)

        # normalize templates to unit sum
        normVec = tensorW.sum(axis=2).sum(axis=0)

        tensorW *= 1.0 / (EPS+np.expand_dims(normVec, axis=1))

    W = list()
    nmfdV = list()

    # compute final output approximation
    for r in range(R):
        W.append(tensorW[:, r, :])
        nmfdV.append(convModel(np.expand_dims(tensorW[:, r, :], axis=1), np.expand_dims(H[r, :], axis=0)))

    return W, H, nmfdV, costFunc, tensorW


def init_parameters(parameter, paramConstr):
    """Auxiliary function to set the parameter dictionary

       Parameters
       ----------
       parameter: dict
           See the above function NMFD for further information

       Returns
       -------
       parameter: dict
    """
    paramConstr = {'type': 'none'} if not paramConstr else paramConstr

    parameter['numComp'] = 3 if 'numComp' not in parameter else parameter['numComp']
    parameter['numIter'] = 30 if 'numIter' not in parameter else parameter['numIter']
    parameter['numTemplateFrames'] = 8 if 'numTemplateFrames' not in parameter else parameter['numTemplateFrames']
    parameter['initW'] = initTemplates(parameter, 'random') if 'initW' not in parameter else parameter['initW']
    parameter['initH'] = initActivations(parameter, 'uniform') if 'initH' not in parameter else parameter['initH']
    parameter['fixH'] = False if 'fixH' not in parameter else parameter['fixH']
    parameter['fixW'] = False if 'fixW' not in parameter else parameter['fixW']

    return parameter, paramConstr
