"""
    Name: drumSpecificSoftConstraintsNMF
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

from scipy.ndimage import median_filter

from NMFtoolbox.percussivenessEstimation import percussivenessEstimation
from NMFtoolbox.NEMA import NEMA


def drumSpecificSoftConstraintsNMF(W, H, iter, numIter, parameter):
    """Implements the drum specific soft constraints that can be applied during
    NMF or NMFD iterations. These constraints affect the activation vectors only and
    are described in sec.23 of [2].

    References
    ----------
    [2] Christian Dittmar, Patricio Lopez-Serrano, Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP), 2018.

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
        Kern     Order of smoothing operation
        Kern     Concrete smoothing kernel
        initH    Initial version of the NMF activations
        initW    Initial version of the NMF templates

    Returns
    -------
    W: array-like
        Processed NMF templates

    H_out: array-like
        Processed NMF activations
    """
    # this assumes that the templates are constructed as described in sec. 2.4 of [2]
    percWeight = percussivenessEstimation(W).reshape(1, -1)

    # promote harmonic sustained gains
    Hh = median_filter(H, footprint=parameter['Kern'], mode='constant')

    # promote decaying impulses gains
    Hp = NEMA(H, parameter['decay'])

    # make weighted sum according to percussiveness measure
    H_out = Hh * (1 - percWeight.T) + Hp * percWeight.T

    return W, H_out
