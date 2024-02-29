"""
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
Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

'NMF toolbox' is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""
import numpy as np
from numpy.finfo import eps as EPS
from tqdm import tnrange
from typing import Tuple, List


def nmf(V,
        num_comp: int,
        cost_func: str = 'KLDiv',
        num_iter: int = 30,
        init_W: np.ndarray = None,
        init_H: np.ndarray = None,
        fix_W: bool = False) -> Tuple[np.ndarray, np.ndarray, List]:
    """Given a non-negative matrix V, find non-negative templates W and activations
    H that approximate V.

    References
    ----------
    [1] Lee, DD & Seung, HS. "Algorithms for Non-negative Matrix Factorization"

    [2] Andrzej Cichocki, Rafal Zdunek, Anh Huy Phan, and Shunichi Amari  Nonnegative Matrix and Tensor Factorizations:
    Applications to Exploratory Multi-Way Data Analysis and Blind Source Separation" John Wiley and Sons, 2009.

    Parameters
    ----------
    V: np.ndarray
        K x M non-negative matrix to be factorized
    num_comp: int
        The rank of the approximation
    cost_func: str
        Cost function used for the optimization, currently supported are:
            'EucDist' for Euclidean Distance
            'KLDiv' for Kullback Leibler Divergence
            'ISDiv' for Itakura Saito Divergence
    num_iter: int
        Number of iterations the algorithm will run.
    init_W: np.ndarray

    init_H: np.ndarray


    fix_W: bool

    Returns
    -------
    W: np.ndarray
        K x R non-negative templates
    H: np.ndarray
        R x M non-negative activations
    nmfV: list
        Approximated component matrices
    """
    # get important params
    K, M = V.shape
    R = num_comp
    L = num_iter

    # initialization of W and H
    W = init_W.copy() if init_W is not None else np.random.rand(K, R)
    H = init_H.copy() if init_H is not None else np.random.rand(R, M)

    # create helper matrix of all ones
    ones_mat = np.ones((K, M))

    # normalize to unit sum
    V /= (EPS + V.sum())

    # main iterations
    for _ in tnrange(L, desc='Processing'):

        # compute approximation
        lamb = EPS + W @ H

        # switch between pre-defined update rules
        if cost_func == 'EucDist':  # euclidean update rules
            if not fix_W:
                W *= (V @ H.T / (lamb @ H.T + EPS))

            H *= (W.T @ V / (W.T @ lamb + EPS))

        elif cost_func == 'KLDiv':  # Kullback Leibler divergence update rules
            if not fix_W:
                W *= ((V / lamb) @ H.T) / (ones_mat @ H.T + EPS)

            H *= (W.T @ (V / lamb)) / (W.T @ ones_mat + EPS)

        elif cost_func == 'ISDiv':  # Itakura Saito divergence update rules
            if not fix_W:
                W *= ((lamb ** -2 * V) @ H.T) / ((lamb ** -1) @ H.T + EPS)

            H *= (W.T @(lamb ** -2 * V)) / (W.T @ (lamb ** -1) + EPS)

        else:
            raise ValueError('Unknown cost function')

        # normalize templates to unit sum
        if not fix_W:
            normVec = W.sum(axis=0)
            W *= 1.0 / (EPS + normVec)

    nmfV = list()

    # compute final output approximation
    for r in range(R):
        nmfV.append(W[:, r].reshape(-1, 1) @ H[r, :].reshape(1, -1))

    return W, H, nmfV
