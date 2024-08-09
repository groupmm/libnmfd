import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from typing import List, Tuple, Union

from libnmfd.dsp.filters import nema
from libnmfd.utils.dsp_utils import conv2
from libnmfd.dsp.transforms import log_freq_log_mag
from . import EPS


def drum_specific_soft_constraints_nmf(W: np.ndarray,
                                       H: np.ndarray,
                                       decay: Union[np.ndarray, float],
                                       kern: int) -> Tuple[np.ndarray, np.ndarray]:
    """Implements the drum specific soft constraints that can be applied during
    NMF or NMFD iterations. These constraints affect the activation vectors only and
    are described in sec.23 of [1].

    References
    ----------
    [1] Christian Dittmar, Patricio López-Serrano, and Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: np.ndarray
        NMF templates given in matrix/tensor form

    H: np.ndarray
        NMF activations given as matrix

    decay: list of np.ndarray
        Optional list of decay values per component.

    kern: int
        Width of the smoothing kernel

    Returns
    -------
    W: np.ndarray
        Processed NMF templates

    H_out: np.ndarray
        Processed NMF activations
    """
    # this assumes that the templates are constructed as described in sec. 2.4 of [2]
    percWeight = percussiveness_estimation(W).reshape(1, -1)

    # promote harmonic sustained gains
    Hh = median_filter(H, footprint=kern, mode='constant')

    # promote decaying impulses gains
    Hp = nema(H, decay)

    # make weighted sum according to percussiveness measure
    H_out = Hh * (1 - percWeight.T) + Hp * percWeight.T

    return W, H_out


def diagonality_soft_constraints_nmf(H: np.ndarray,
                                     kern_ord: int) -> np.ndarray:
    """Implements a simplified version of the soft constraints in [1]

    References
    ----------
    [1] Jonathan Driedger, Thomas Prätzlich, and Meinard Müller
    Let It Bee -- Towards NMF-Inspired Audio Mosaicing
    In Proceedings of the International Conference on Music Information Retrieval (ISMIR): 350-356, 2015.

    Parameters
    ----------
    H: np.ndarray
        NMF activations given as matrix

    kern_ord: int
        Order of smoothing operation

    Returns
    -------
    H: np.ndarray
        Processed NMF activations
    """

    H = conv2(H, np.eye(kern_ord), 'same')

    return H


def percussiveness_estimation(W: np.ndarray) -> np.ndarray:
    """This function takes a matrix or tensor of NMF templates and estimates the percussiveness by assuming that the
    lower part explains percussive and the upper part explains harmonic components. This is explained in sec. 2.4,
    especially eq. (4) in [1].

    References
    ----------
    [1] Christian Dittmar, Patricio López-Serrano, and Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: np.ndarray
        K x R matrix (or K x R x T tensor) of NMF (NMFD) templates

    Returns
    -------
    perc_weight: np.ndarray
        The resulting percussiveness estimate per component
    """
    # get dimensions of templates
    K, R, T = W.shape

    # this assumes that the matrix (tensor) is formed in the way we need it
    num_bins = int(K/2)

    # do the calculation, which is essentially a ratio
    perc_weight = np.zeros(R)

    for c in range(R):
        perc_part = W[:num_bins, c, :]
        harm_part = W[:, c, :]
        perc_weight[c] = perc_part.sum() / harm_part.sum()

    return perc_weight


def colored_components(comp_A,
                       col_vec=None):
    """Maps a list containing parallel component spectrograms into a color-coded spectrogram image, similar to Fig. 10
    in [1]. Works best for three components corresponding to RGB.

    References
    ----------
    [1] Christian Dittmar and Meinard Müller
    "Reverse Engineering the Amen Break - Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531-1543, 2016.

    Parameters
    ----------
    comp_A: list
        List with the component spectrograms, all should have the same dimensions

    col_vec: list
        List with color codes given externally, if not provided some defaults will be used

    Returns
    -------
    rgbA: np.ndarray
        Color-coded spectrogram
    """

    num_comp = len(comp_A)
    num_bins, num_frames = comp_A[0].shape
    color_slice = np.zeros((num_bins, num_frames, 3))

    if col_vec is not None:
        col_vec = rgb_to_hsv(col_vec)
    else:
        if num_comp == 1:
            pass
        elif num_comp == 2:
            col_vec = [[1, 0, 0], [0, 1, 1]]
        elif num_comp == 3:
            col_vec = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif num_comp == 4:
            col_vec = [[1, 0, 0], [0.5, 1, 0], [0, 1, 1], [0.5, 0, 1]]
        else:
            col_vec = [to_rgb(cm.hsv(i * 1 / num_comp, 1)) for i in range(0, num_comp)]

    rgb_A = np.zeros((num_bins, num_frames, 3))

    for k in range(num_comp):
        maxVal = comp_A[k].max()

        if maxVal < EPS:
            maxVal = 1.0

        intensity = 1 - comp_A[k] / maxVal

        for g in range(3):
            color_slice[:, :, g] = col_vec[k][g] * intensity

        rgb_A += color_slice

    # convert to HSV space
    hsv_A = rgb_to_hsv(rgb_A)

    # invert luminance
    hsv_A[:, :, 2] /= hsv_A[:, :, 2].max(0).max(0)

    # shift hue circularly
    hsv_A[:, :, 0] = np.mod((1/(num_comp-1)) + hsv_A[:, :, 0], 1)

    # convert to RGB space
    rgb_A = hsv_to_rgb(hsv_A)

    return rgb_A


def visualize_components_kam(comp_A: List,
                             time_res: float,
                             freq_res: float,
                             start_sec: float = None,
                             end_sec: float = None,
                             font_size: float = 11) -> matplotlib.figure.Figure:
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components, this function provides a
    visualization.

    Parameters
    ----------
    comp_A: list
        List with R individual component magnitude spectrograms.

    time_res: float
        Temporal resolution

    freq_res: float
        Spectral resolution

    start_sec: float
        Where to zoom in on the time axis

    end_sec: float
        Where to zoom in on the time axis

    font_size: float
        Font size of the figure.

    Returns
    -------
    fh: matplotlib.figure.Figure
        The figure handle
    """
    # get spectrogram dimensions
    num_lin_bins, num_frames = comp_A[0].shape

    start_sec = time_res if start_sec is None else start_sec
    end_sec = num_frames * time_res if end_sec is None else end_sec

    # plot MMF / NMFD components
    # map template spectrograms to a logarithmically - spaced frequency
    # and logarithmic magnitude compression

    log_freq_log_mag_comp_A, log_freq_axis = log_freq_log_mag(A=comp_A, freq_res=freq_res)
    num_log_bins = len(log_freq_axis)

    time_axis = np.arange(num_frames) * time_res
    freq_axis = np.arange(num_log_bins)

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': font_size}
    matplotlib.rc('font', **font)

    # make new  figure
    fh, ax = plt.subplots(figsize=(15, 10))

    # plot the component spectrogram matrix
    ax.imshow(colored_components(log_freq_log_mag_comp_A),
              origin='lower', aspect='auto', cmap='gray_r',
              extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

    ax.set_xlim(start_sec, end_sec)
    ax.set_title('A = A_p + A_h')
    ax.set_xlabel('Time in seconds')
    ax.set_yticklabels([])

    return fh


def visualize_components_nmf(V: np.ndarray,
                             W: np.ndarray,
                             H: np.ndarray,
                             comp_V: np.ndarray,
                             log_comp: float = 1.0,
                             time_res: float = None,
                             freq_res: np.ndarray = None,
                             start_sec: float = None,
                             end_sec: float = None,
                             font_size: float = 11) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components, this function provides a
    visualization.

    Parameters
    ----------
    V: np.ndarray
        K x M non-negative target matrix, in our case, this is usually a magnitude spectrogram

    W: np.ndarray
        K X R matrix of learned template matrices

    H: np.ndarray
        R X M matrix of learned activations

    comp_V: np.ndarray
        Matrix with R individual component magnitude spectrograms

    log_comp: float
        Factor to control the logarithmic magnitude compression

    time_res: float
        Temporal resolution

    freq_res: float
        Spectral resolution

    start_sec: float
        Where to zoom in on the time axis

    end_sec: float
        Where to zoom in on the time axis

    font_size: float
        Font size of the figure

    Returns
    -------
    fh: matplotlib.figure.Figure
        The figure handle

    log_freq_axis: np.ndarray
        Log frequency axis
    """
    R = H.shape[0]
    num_lin_bins, num_frames = V.shape
    comp_col_vec = __set_comp_vol_vec(R)

    # plot MMF / NMFD components
    # map the target and the templates to a logarithmically-spaced frequency
    # and logarithmic magnitude compression
    log_freq_log_magV, log_freq_axis = log_freq_log_mag(V, freq_res=freq_res, log_comp=log_comp)
    num_log_bins = len(log_freq_axis)

    log_freq_log_mag_W, log_freq_axis = log_freq_log_mag(W, freq_res=freq_res, log_comp=log_comp)

    if comp_V is not None:
        log_freq_log_mag_comp_V, log_freq_axis = log_freq_log_mag(A=comp_V, freq_res=freq_res, log_comp=log_comp)
    else:
        log_freq_log_mag_comp_V = [np.array(log_freq_log_magV)]  # simulate one component

    time_axis = np.arange(num_frames) * time_res
    freq_axis = np.arange(num_log_bins)

    # subsample freq axis, this is mainly for visualization purposes to have
    # ticks only at the positions of multiples / divisors of 440.0 Hz
    sub_samp = np.where(np.mod(log_freq_axis.astype(np.float32), 55.0) < 0.001)[0]
    sub_samp_freq_axis = log_freq_axis[np.mod(log_freq_axis.astype(np.float32), 55.0) < 0.001]

    font = {'family': 'sans-serif',
            'weight': 'normal',
            'size': font_size}

    matplotlib.rc('font', **font)

    # normalize NMF / NMFD activations to unit maximum
    H *= 1 / (EPS + np.max(H.T, axis=0).reshape(-1, 1))

    fh = plt.figure(constrained_layout=False, figsize=(20, 20))
    gs = fh.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1])
    ax1 = fh.add_subplot(gs[1, 1])

    # first, plot the component spectrogram matrix
    if R <= 4 or len(log_freq_log_mag_comp_V) == 2:
        ax1.imshow(colored_components(log_freq_log_mag_comp_V), origin='lower', aspect='auto', cmap='gray_r',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

    else:
        ax1.imshow(log_freq_log_magV, origin='lower', aspect='auto', cmap='gray_r',
                   extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])

    ax1.set_xlabel('Time in seconds')
    ax1.set_xlim(start_sec, end_sec)
    ax1.set_yticklabels([])

    # second, plot the activations as polygons
    # decide between different visualizations
    ax2 = fh.add_subplot(gs[0, 1])

    if R > 10:
        ax2.imshow(H, origin='lower', aspect='auto', cmap='gray_r', extent=[time_axis[0], time_axis[-1], 1, R])
        ax2.set_ylabel('Template')
        ax2.set_yticks([1, R])
        ax2.set_yticklabels([0, R-1])
    else:

        for r in range(R):
            curr_activation = 0.95 * H[r, :]  # put some  headroom
            xcoord = 0.5 / freq_res + np.concatenate([time_axis.reshape(1, -1), np.fliplr(time_axis.reshape(1, -1))], axis=1)
            ycoord = r + np.concatenate([np.zeros((1, num_frames)), np.fliplr(curr_activation.reshape(1, -1))], axis=1)
            ax2.fill(xcoord.squeeze(), ycoord.squeeze(), color=comp_col_vec[r, :])
            ax2.set_ylim(0, R)
            ax2.set_yticks(0.5 + np.arange(0, R))
            ax2.set_yticklabels(np.arange(1, R+1))

    ax2.set_xlim(start_sec, end_sec)

    # third, plot the templates
    if R > 10:
        ax3 = fh.add_subplot(gs[1, 0])
        num_template_frames = 1
        if isinstance(log_freq_log_mag_W, list):
            num_template_frames = log_freq_log_mag_W[0].shape[1]
            norm_W = np.concatenate(log_freq_log_mag_W,axis=1)
        else:
            norm_W = log_freq_log_mag_W.copy()

        norm_W *= 1 / (EPS + norm_W.max(axis=0))

        ax3.imshow(norm_W, aspect='auto', cmap='gray_r', origin='lower',
                   extent=[0, (R*num_template_frames)-1, sub_samp_freq_axis[0], sub_samp_freq_axis[-1]])
        ax3.set_xticks([0, R*num_template_frames])
        ax3.set_xticklabels([0, R-1])

        ax3.set_xlabel('Template')
        ax3.set_ylabel('Frequency in Hz')

    else:
        axs3 = list()
        for r in range(R):
            gs3 = gs[1, 0].subgridspec(nrows=1, ncols=R, hspace=0)
            axs3.append(fh.add_subplot(gs3[0, r]))

            if isinstance(log_freq_log_mag_W, list):
                curr_template = np.array(log_freq_log_mag_W[r])
            else:
                curr_template = log_freq_log_mag_W[:, r].copy()

            temp_list = list()
            if R <= 4:
                # make a trick to color code the template spectrograms
                for g in range(R):
                    temp_list.append(np.zeros(curr_template.shape))

                temp_list[r] = curr_template
                axs3[r].imshow(colored_components(temp_list), origin='lower', aspect='auto')

            else:
                curr_template /= curr_template.max(axis=0)
                axs3[r].imshow(curr_template, origin='lower', aspect='auto', cmap='gray_r')

            if r == 0:
                axs3[r].set_yticks(sub_samp)
                axs3[r].set_yticklabels(np.round(sub_samp_freq_axis))
                axs3[r].set_ylabel('Frequency in Hz')

            else:
                axs3[r].set_yticklabels([])
            axs3[r].set_xticklabels([])
            axs3[r].set_xlabel(str(r+1))

    return fh, log_freq_axis


def __set_comp_vol_vec(R):
    if R == 2:
        return np.array([[1, 0, 0], [0, 0.5, 0.5]], dtype=float)
    elif R == 3:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elif R == 4:
        return np.array([[1, 0, 1], [1, 0.5, 0], [0, 1, 0], [0, 0.5, 1]], dtype=float)
    else:
        return np.tile(np.array([0.5, 0.5, 0.5]), (R, 1)).astype(float)
