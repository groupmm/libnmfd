"""
    Name: NMFdiag
    Date of Revision: Jun 2019
    Programmer: Jonathan Driedger, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

from copy import deepcopy
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from tqdm import tnrange

from NMFtoolbox.utils import EPS


def NMFdiag(V, W0, H0, parameter=None):
    """Given a non-negative matrix V, find non-negative matrix factors W and H
    such that V ~ WH. Possibly also enforce continuity constraints.

    References
    ----------
    [2] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"
    Ewert, S. & Mueller, M. "Using Score-Informed Constraints For NMF-Based
    Source Separation"

    Parameters
    ----------
    V: array-like
        NxM matrix to be factorized

    W0: array-like
        Initialized W matrix

    H0: array-like
        Initialized H matrix

    parameter: dict
        distMeas    Distance measure which is used for the optimization.
                    Values are 'euclidean' for Euclidean, or 'divergence'
                    for KL-divergence.
        numOfIter   Number of iterations the algorithm will run.
        fixW        Set to 1 if Templates W should be fixed during the
                    update process.
                    divergence cost function update rules # TODO: ?
        continuity  Set of parameters related to the enforced continuity
                    constraints.
        length      Number of templates which should be activated
                    successively.
        decay       Parameter for specifying the decaying gain of successively
                    activated templates.
        grid        This value indicates in wich iterations of the NMF update
                    procedure the continuity constraints should be enforced.

    Returns
    -------
    W: array-like
        NxK non-negative matrix factor

    H: array-like
        KxM non-negative matrix factor
    """
    N, M = V.shape  # V matrix dimensions

    parameter = init_parameters(parameter)
    assertions(V, W0, H0)

    numOfSimulAct = parameter['continuity']['polyphony']
    vis = parameter['vis']

    # V matrix factorization
    #  initialization of W and H
    W = deepcopy(W0)
    H = deepcopy(H0)

    fixW = parameter['fixW']

    energyInW = np.sum(W**2, axis=0).reshape(-1, 1)
    energyScaler = np.tile(energyInW, (1, H.shape[1]))

    # prepare the max neighborhood kernel
    s = np.array(parameter['continuity']['sparsen'])
    assert np.mod(s[0], 2) == 1 and np.mod(s[1], 2) == 1, 'Sparsity parameter needs to be odd!'

    maxFiltKernel = np.zeros(s)
    maxFiltKernel[:, np.ceil(s[0] / 2).astype(int) - 1] = 1
    maxFiltKernel[np.ceil(s[0] / 2).astype(int) - 1, :] = 1

    for k in tnrange(parameter['numOfIter'], desc='Processing'):
        if vis:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.imshow(H, aspect='auto', cmap='gray_r')
            ax.set_title('Activation Matrix H in Iteration {}'.format(k+1))

        # in every 'grid' iteration of the update...
        if np.mod(k, parameter['continuity']['grid']) == 0:

            # sparsen the activations
            if s.max() > 1:

                # should in principle also include the energyScaler...
                H_filt = maximum_filter(H, footprint=maxFiltKernel, mode='constant')  # find max values in neighborhood

                cond = np.array(H != np.array(H_filt))
                H = np.where(cond, H * (1 - (k + 1) / parameter['numOfIter']), H)

            # ...restrict polyphony...
            if numOfSimulAct < H.shape[1]:
                sortVec = np.argsort(np.multiply(-H, energyScaler), axis=0)

                for j in range(H.shape[1]):
                    H[sortVec[numOfSimulAct:, j], j] *= (1 - (k + 1) / parameter['numOfIter'])

            # ... and enforce continuity
            filt = np.eye(parameter['continuity']['length'])
            H = convolve2d(H, filt, 'same')

        if parameter['distMeas'] == 'euclidean':  # euclidean update rules
            H *= (W.T @ V) / (W.T @ W @ H + EPS)

            if not fixW:
                W *= (V @ H.T / ((W @ H @ H.T) + EPS))

        elif parameter['distMeas'] == 'divergence':  # divergence update rules
            H *= (W.T @ (V / (W @ H + EPS))) / (np.sum(W, axis=0).T.reshape(-1, 1) @ np.ones((1, M)) + EPS)

            if not fixW:
                W *= ((V / (W @ H + EPS)) @ H.T) / (np.ones((N, 1)) @ np.sum(H, axis=1).reshape(1, -1) + EPS)

        else:
            raise ValueError('Unknown distance measure')

    if vis:
        _, ax2 = plt.subplots(figsize=(15, 10))
        ax2.imshow(H, aspect='auto', cmap='gray_r')
        ax2.set_title('Final Activation Matrix H')

    return W, H


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

       Parameters
       ----------
       parameter: dict
           See the above function NMFdiag for further information

       Returns
       -------
       parameter: dict
    """
    parameter = dict() if parameter is None else parameter
    parameter['distMeas'] = 'divergence' if 'distMeas' not in parameter else parameter['distMeas']
    parameter['numOfIter'] = 50 if 'fixW' not in parameter else parameter['numOfIter']
    parameter['fixW'] = False if 'fixW' not in parameter else parameter['fixW']
    parameter['continuity'] = {'length': 10,
                               'grid': 5,
                               'sparsen': [1, 1],
                               'polyphony': 5} if 'continuity' not in parameter else parameter['continuity']

    parameter['vis'] = False if 'vis' not in parameter else parameter['vis']

    return parameter


def assertions(V, W0, H0):
    """
        Auxiliary function to throw assertion errors, if needed

        Parameters
        ----------
        V: array-like
            NxM matrix to be factorized
        W0: array-like
            Initialized W matrix
        H0: array-like
            Initialized H matrix
    """
    # matrices dimensions consistency check
    N, M = V.shape  # V matrix dimensions
    WN, WK = W0.shape  # W matrix dimensions
    HK, HM = H0.shape  # H matrix dimensions
    K = deepcopy(WK)

    # check if W matrix dimensions are consistent
    assert WN == N, 'W matrix has inconsistent dimensions.'

    # check if H matrix dimensions are consistent
    assert HK == K and HM == M, 'H matrix has inconsistent dimensions.'

    # matrices non-negativity check

    #  check V matrix is non-negative
    assert not np.any(V < 0), 'V matrix must not contain negative values.'

    #  check W0 matrix is non-negative
    assert not np.any(W0 < 0), 'W0 matrix must not contain negative values.'

    #  check H0 matrix is non-negative
    assert not np.any(H0 < 0), 'H0 matrix must not contain negative values.'
