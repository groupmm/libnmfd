"""
    Name: diagonalitySoftConstraintsNMF
    Date: Jun 2019
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

import numpy as np
from NMFtoolbox.utils import conv2


def diagonalitySoftConstraintsNMF(W, H, iter, numIter, parameter):
    """Implements a simplified version of the soft constraints in [2].
    Name: diagonalitySoftConstraintsNMF
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    References:
    [2] Jonathan Driedger, Thomas Prätzlich, and Meinard Müller
    Let It Bee -- Towards NMF-Inspired Audio Mosaicing
    In Proceedings of the International Conference on Music Information
    Retrieval (ISMIR): 350-356, 2015.

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
        KernOrd:    Order of smoothing operation
        initW:      Initial version of the NMF templates

    Returns
    -------
    W: array-like
        Processed NMF templates

    H: array-like
        Processed NMF activations
    """

    H = conv2(H, np.eye(parameter['KernOrd']), 'same')

    return W, H