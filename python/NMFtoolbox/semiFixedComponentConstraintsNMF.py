"""
    Name: semiFixedComponentConstraintsNMF
    Date: Jun 2018
    Programmer: Christian Dittmar, Patricio López-Serrano, Yiğitcan Özer

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
from copy import deepcopy


def semiFixedComponentConstraintsNMF(W, H, iter, numIter, parameter):
    """Implements a simplified version of the soft constraints in [2].

    References
    ----------
    [2] Patricio López-Serrano, Christian Dittmar, Jonathan Driedger, and
        Meinard Müller.
        Towards modeling and decomposing loop-based
        electronic music.
        In Proceedings of the International Conference
        on Music Information Retrieval (ISMIR), pages 502–508,
        New York City, USA, August 2016.

    [3] Christian Dittmar and Daniel Gärtner
        Real-time transcription and separation of drum recordings based on
        NMF decomposition.
        In Proceedings of the International Conference on Digital Audio
        Effects (DAFx): 187–194, 2014.

    Parameters
    ----------
    W: array-like
        NMF templates given in matrix/tensor form

    H: array-like
        NMF activations given as matrix

    iter: int
        Current iteration count

    numIter: int
        Target number of iterations

    parameter: dict
        initW:            Initial version of the NMF templates
        initH:            Initial version of the NMF activations
        adaptDegree:      0 = fixed, 1 = semi-fixed, 2 = adaptive
        adaptTiming:      Only in combination with semi-fixed adaptDegree

    Returns
    -------
    W: array-like
        processed NMF templates

    H: array-like
        processed NMF activations
    """
    # roughly equal to Driedger
    # mix with originally estimated sources
    numRows = W.shape[0]

    weight = (iter + 1.0) / numIter

    if not isinstance(parameter['initW'], list):
        newWeight = np.tile(weight * parameter['adaptDegree'] ** parameter['adaptTiming'], (numRows, 1))
        initW = deepcopy(parameter['initW'])

    else:
        R = len(parameter['initW'])
        newWeight = np.zeros(W.shape)

        initW = np.zeros(W.shape)

        # stack the templates into a tensor
        for r in range(R):
            initW[:, r, :] = parameter['initW'][r]
            cMat = np.tile(weight * parameter['adaptDegree'][r] ** parameter['adaptTiming'][r],
                           (numRows, W.shape[2]))
            newWeight[:, r, :] = cMat

    newWeight[:, parameter['adaptDegree'].squeeze() > 1.0] = 1

    # constrain the range of values
    newWeight = np.minimum(newWeight, 1.0)
    newWeight = np.maximum(newWeight, 0.0)

    if newWeight.shape != W.shape:
        newWeight = np.expand_dims(newWeight, axis=2)

    if initW.shape != W.shape:
        initW = np.expand_dims(initW, axis=2)

    # compute the counter part
    initWeight = 1 - newWeight

    W = W * newWeight + initW * initWeight

    return W, H
