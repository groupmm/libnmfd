"""
    Name: visualizeComponentsNMF
    Date: Aug 2019
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
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy

from NMFtoolbox.logFreqLogMag import logFreqLogMag
from NMFtoolbox.utils import EPS
from NMFtoolbox.coloredComponents import coloredComponents


def visualizeComponentsNMF(V, W, H, compV=None, parameter=None):
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components,
    this function provides a visualization.

    Parameters
    ----------
    V: array-like
        K x M non-negative target matrix, in our case, this is usually
        a magnitude spectrogram

    W: array-like
        Numpy array with R indiviual K X T learned template matrices

    H: array-like
        R X M matrix of learned activations

    compV: array-like
        numpy array with R individual component magnitude spectrograms

    parameter: dict
        deltaT        Temporal resolution
        deltaF        Spectral resolution
        startSec      Where to zoom in on the time axis
        endeSec       Where to zoom in on the time axis

    Returns
    -------
    fh: The figure handle
    """
    R = H.shape[0]
    numLinBins, numFrames = V.shape

    parameter = init_params(H, V, parameter)

    # plot MMF / NMFD components

    # map the target and the templates to a logarithmically-spaced frequency
    # and logarithmic magnitude compression
    logFreqLogMagV, logFreqAxis = logFreqLogMag(V, parameter['deltaF'], logComp=parameter['logComp'])
    numLogBins = len(logFreqAxis)

    logFreqLogMagW, logFreqAxis = logFreqLogMag(W, parameter['deltaF'], logComp=parameter['logComp'])

    if compV is not None:
        logFreqLogMagCompV, logFreqAxis = logFreqLogMag(compV, parameter['deltaF'], logComp=parameter['logComp'])
    else:
        logFreqLogMagCompV = [np.array(logFreqLogMagV)]  # simulate one component

    timeAxis = np.arange(numFrames) * parameter['deltaT']
    freqAxis = np.arange(numLogBins)

    # subsample freq axis
    subSamp = np.where(np.mod(logFreqAxis.astype(np.float32), 55) < 0.001)[0]
    subSampFreqAxis = logFreqAxis[np.mod(logFreqAxis.astype(np.float32), 55) < 0.001]

    # further plot params
    setFontSize = 11 if 'fontSize' not in parameter else parameter['fontSize']

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': setFontSize}
    matplotlib.rc('font', **font)

    startSec = parameter['startSec']
    endeSec = parameter['endeSec']

    # normalize NMF / NMFD activations to unit maximum
    H *= 1 / (EPS + np.max(H.T, axis=0).reshape(-1, 1))

    fh = plt.figure(constrained_layout=False, figsize=(20, 20))
    gs = fh.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])
    ax1 = fh.add_subplot(gs[1, 1])

    # first, plot the component spectrogram matrix
    if R <= 4 or len(logFreqLogMagCompV) == 2:
        ax1.imshow(coloredComponents(logFreqLogMagCompV), origin='lower', aspect='auto', cmap='gray_r',
                   extent=[timeAxis[0], timeAxis[-1], freqAxis[0], freqAxis[-1]])

    else:
        ax1.imshow(logFreqLogMagV, origin='lower', aspect='auto', cmap='gray_r',
                   extent=[timeAxis[0], timeAxis[-1], freqAxis[0], freqAxis[-1]])

    ax1.set_xlabel('Time in seconds')
    ax1.set_xlim(startSec, endeSec)
    ax1.set_yticklabels([])

    # second, plot the activations as polygons
    # decide between different visualizations
    ax2 = fh.add_subplot(gs[0, 1])

    if R > 10:
        ax2.imshow(H, origin='lower', aspect='auto', cmap='gray_r', extent=[timeAxis[0], timeAxis[-1], 1, R])
        ax2.set_ylabel('Template')
        ax2.set_yticks([1, R])
        ax2.set_yticklabels([0, R-1])
    else:

        for r in range(R):
            currActivation = 0.95 * H[r, :]  # put some  headroom
            xcoord = 0.5 / parameter['deltaF'] + np.concatenate([timeAxis.reshape(1, -1), np.fliplr(timeAxis.reshape(1, -1))], axis=1)
            ycoord = r + np.concatenate([np.zeros((1, numFrames)), np.fliplr(currActivation.reshape(1, -1))], axis=1)
            ax2.fill(xcoord.squeeze(), ycoord.squeeze(), color=parameter['compColVec'][r, :])
            ax2.set_ylim(0, R)
            ax2.set_yticks(0.5 + np.arange(0, R))
            ax2.set_yticklabels(np.arange(1, R+1))

    ax2.set_xlim(startSec, endeSec)

    # third, plot the templates
    if R > 10:
        ax3 = fh.add_subplot(gs[1, 0])
        numTemplateFrames = 1
        if isinstance(logFreqLogMagW, list):
            numTemplateFrames = logFreqLogMagW[0].shape[1]
            normW = np.concatenate(logFreqLogMagW,axis=1)
        else:
            normW = deepcopy(logFreqLogMagW)

        normW *= 1 / (EPS + normW.max(axis=0))

        ax3.imshow(normW, aspect='auto', cmap='gray_r', origin='lower',
                   extent=[0, (R*numTemplateFrames)-1, subSampFreqAxis[0], subSampFreqAxis[-1]])

        ax3.set_xticks([0, R*numTemplateFrames])
        ax3.set_xticklabels([0, R-1])

        ax3.set_xlabel('Template')
        ax3.set_ylabel('Frequency in Hz')

    else:
        axs3 = list()
        for r in range(R):
            gs3 = gs[1, 0].subgridspec(nrows=1, ncols=R, hspace=0)
            axs3.append(fh.add_subplot(gs3[0, r]))

            if isinstance(logFreqLogMagW, list):
                currTemplate = np.array(logFreqLogMagW[r])
            else:
                currTemplate = deepcopy(logFreqLogMagW[:, r])

            temp_list = list()
            if R <= 4:
                # make a trick to color code the template spectrograms
                for g in range(R):
                    temp_list.append(np.zeros(currTemplate.shape))

                temp_list[r] = currTemplate
                axs3[r].imshow(coloredComponents(temp_list), origin='lower', aspect='auto')

            else:
                currTemplate /= currTemplate.max(axis=0)
                axs3[r].imshow(currTemplate, origin='lower', aspect='auto', cmap='gray_r')

            if r == 0:
                axs3[r].set_yticks(subSamp)
                axs3[r].set_yticklabels(np.round(subSampFreqAxis))
                axs3[r].set_ylabel('Frequency in Hz')

            else:
                axs3[r].set_yticklabels([])
            axs3[r].set_xticklabels([])
            axs3[r].set_xlabel(str(r+1))

    return fh, logFreqAxis


def init_params(H, V, parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function visualizeComponentsNMF for further information

    Returns
    -------
    parameter: dict
    """
    R = H.shape[0]
    numLinBins, numFrames = V.shape

    if not 'compColVec' in parameter:
        parameter['compColVec'] = dict()

        if R == 2:
            parameter['compColVec'] = np.array([[1, 0, 0], [0, 0.5, 0.5]], dtype=np.float)

        elif R == 3:
            parameter['compColVec'] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)

        elif R == 4:
            parameter['compColVec'] = np.array([[1, 0, 1], [1, 0.5, 0], [0, 1, 0], [0, 0.5, 1]], dtype=np.float)

        else:
            parameter['compColVec'] = np.tile(np.array([0.5, 0.5, 0.5]), (R, 1)).astype(np.float)

    parameter['startSec'] = parameter['deltaT'] if 'startSec' not in parameter else parameter['startSec']
    parameter['endeSec'] = numFrames*parameter['deltaT'] if 'endeSec' not in parameter else parameter['endeSec']
    parameter['logComp'] = 1.0 if 'logComp' not in parameter else parameter['logComp']

    return parameter
