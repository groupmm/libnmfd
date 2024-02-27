"""
    Name: percussivenessEstimation
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

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


import numpy as np


def percussivenessEstimation(W):
    """This function takes a matrix or tensor of NMF templates and estimates the
    percussiveness by assuming that the lower part explains percussive and the
    upper part explains harmonic components. This is explained in sec. 2.4,
    especially eq. (4) in [2].

    References
    ----------
    [2] Christian Dittmar, Patricio López-Serrano, Meinard Müller: "Unifying
    Local and Global Methods for Harmonic-Percussive Source Separation"
    In Proceedings of the IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: array-like
        K x R matrix (or K x R x T tensor) of NMF (NMFD) templates

    Returns
    -------
    percWeight: array-like
        The resulting percussiveness estimate per component
    """
    # get dimensions of templates
    K, R, T = W.shape

    # this assumes that the matrix (tensor) is formed in the way we need it
    numBins = int(K/2)

    # do the calculation, which is essentially a ratio
    percWeight = np.zeros(R)

    for c in range(R):
        percPart = W[:numBins, c, :]
        # harmPart = squeeze(W(1:end,c,:));
        harmPart = W[:, c, :]
        percWeight[c] = percPart.sum() / harmPart.sum()

    return percWeight
