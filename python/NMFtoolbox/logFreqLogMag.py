"""
    Name: logFreqLogMag
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

from NMFtoolbox.midi2freq import midi2freq


def logFreqLogMag(A, deltaF, binsPerOctave=36.0, upperFreq=22050.0, lowerFreq=midi2freq(24), logComp=1.0):
    """Given a magnitude spectrogram, this function maps it onto a compact
    representation with logarithmically spaced frequency axis and logarithmic
    magnitude compression.

    Parameters
    ----------
    A: array-like
        The real-valued magnitude spectrogram oriented as numBins x numFrames,
        it can also be given as a list of multiple spectrograms

    deltaF: array-like
        The spectral resolution of the spectrogram

    binsPerOctave: array-like
        The spectral selectivity of the log-freq axis

    upperFreq: float
        The upper frequency border

    lowerFreq: float
        The lower frequency border

    logComp: float
        Factor to control the logarithmic magnitude compression

    Returns
    -------
    logFreqLogMagA: array-like
        The log-magnitude spectrogram on logarithmically spaced frequency axis

    logFreqAxis: array-like
        An array giving the center frequencies of each bin along the
        logarithmically spaced frequency axis
    """

    # convert to list if necessary
    if not isinstance(A, list):
        wasArrInput = True
        A = [A]
    else:
        wasArrInput = False

    # get number of components
    numComp = len(A)

    logFreqLogMagA = list()

    for k in range(numComp):
        # get component spectrogram
        compA = A[k]

        # get dimensions
        numLinBins, numFrames = compA.shape

        # set up linear frequency axis
        linFreqAxis = np.arange(0, numLinBins) * deltaF

        # get upper limit
        upperFreq = linFreqAxis[-1]

        # set up logarithmic frequency axis
        numLogBins = np.ceil(binsPerOctave * np.log2(upperFreq / lowerFreq))
        logFreqAxis = np.arange(0, numLogBins)
        logFreqAxis = lowerFreq * np.power(2.0, logFreqAxis / binsPerOctave)

        # map to logarithmic axis by means of linear interpolation
        logBinAxis = logFreqAxis / deltaF

        # compute linear interpolation for the logarithmic mapping
        floorBinAxis = np.floor(logBinAxis).astype(np.int) - 1
        ceilBinAxis = floorBinAxis + 1
        fraction = logBinAxis - floorBinAxis - 1

        # get magnitude values
        floorMag = compA[floorBinAxis, :]
        ceilMag = compA[ceilBinAxis, :]

        # compute weighted sum
        logFreqA = floorMag * (1 - fraction).reshape(-1, 1) + ceilMag * fraction.reshape(-1, 1)

        # apply magnitude compression
        logFreqLogMagA.append(np.log(1 + (logComp * logFreqA)))

    # revert back to matrix if necessary
    if wasArrInput:
        logFreqLogMagA = np.array(logFreqLogMagA[0])

    return logFreqLogMagA, logFreqAxis.reshape(-1, 1)
