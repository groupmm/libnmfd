"""
    Name: shiftOperator
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


def shiftOperator(A, shiftAmount):
    """Shift operator as described in eq. (5) from [2]. It shifts the columns
    of a matrix to the left or the right and fills undefined elements with
    zeros.

    References
    ----------
    [2] Paris Smaragdis "Non-negative Matrix Factor Deconvolution;
    Extraction of Multiple Sound Sources from Monophonic Inputs".
    International Congress on Independent Component Analysis and Blind Signal
    Separation (ICA), 2004

    Parameters
    ----------
    A: array-like
        Arbitrary matrix to undergo the shifting operation

    shiftAmount: int
        Positive numbers shift to the right, negative numbers
        shift to the left, zero leaves the matrix unchanged

    Returns
    -------
    shifted: array-like
        Result of this operation
    """
    # Get dimensions
    numRows, numCols = A.shape

    # Limit shift range
    shiftAmount = np.sign(shiftAmount) * min(abs(shiftAmount), numCols)

    # Apply circular shift along the column dimension
    shifted = np.roll(A, shiftAmount, axis=-1)

    if shiftAmount < 0:
        shifted[:, numCols + shiftAmount: numCols] = 0

    elif shiftAmount > 0:
        shifted[:, 0: shiftAmount] = 0

    else:
        pass

    return shifted
