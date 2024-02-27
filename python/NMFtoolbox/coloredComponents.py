"""
    Name: coloredComponents
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
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
import matplotlib.cm as cm

from NMFtoolbox.utils import EPS


def coloredComponents(compA, colVec=None):
    """Maps a list containing parallel component spectrograms into a
    color-coded spectrogram image, similar to Fig. 10 in [1].
    Works best for three components corresponding to RGB.

    References
    ----------
    [2] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    compA: array-like
        List with the component spectrograms, all should have the same dimensions

    colVec:

    Returns
    -------
    rgbA: array-like
        color-coded spectrogram
    """

    numComp = len(compA)
    numBins, numFrames = compA[0].shape
    colorSlice = np.zeros((numBins, numFrames, 3))

    if colVec:
        colVec = rgb_to_hsv(colVec)
    else:
        if numComp == 1:
            pass
        elif numComp == 2:
            colVec = [[1, 0, 0], [0, 1, 1]]
        elif numComp == 3:
            colVec = [[1, 0, 0], [0, 1, 0],  [0, 0, 1]]
        elif numComp == 4:
            colVec = [[1, 0, 0], [0.5, 1, 0],  [0, 1, 1], [0.5, 0, 1]]
        else:
            colVec = [to_rgb(cm.hsv(i * 1 / numComp, 1)) for i in range(0, numComp)]

    rgbA = np.zeros((numBins, numFrames, 3))

    for k in range(numComp):

        maxVal = compA[k].max()
        if maxVal < EPS:
            maxVal = 1.0

        intensity = 1 - compA[k] / maxVal

        for g in range(3):
            colorSlice[:, :, g] = colVec[k][g] * intensity

        rgbA += colorSlice

    # convert to HSV space
    hsvA = rgb_to_hsv(rgbA)

    # invert luminance
    hsvA[:, :, 2] /= hsvA[:, :, 2].max(0).max(0)

    # shift hue circularly
    hsvA[:, :, 0] = np.mod((1/(numComp-1)) + hsvA[:, :, 0], 1)

    # convert to RGB space
    rgbA = hsv_to_rgb(hsvA)

    return rgbA
