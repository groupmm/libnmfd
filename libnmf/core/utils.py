import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb, to_rgb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from typing import Union, List

from libnmf.dsp.utils import conv2
from libnmf.dsp.filters import nema
from libnmf.dsp.transforms import log_freq_log_mag
from libnmf.utils import EPS, load_matlab_dict, midi2freq



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
    [1] Christian Dittmar, Patricio López-Serrano, Meinard Müller:
    "Unifying Local and Global Methods for Harmonic-Percussive Source Separation"
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




def drum_specific_soft_constraints_nmf(W: np.ndarray,
                                       H: np.ndarray,
                                       decay,
                                       parameter):
    """Implements the drum specific soft constraints that can be applied during
    NMF or NMFD iterations. These constraints affect the activation vectors only and
    are described in sec.23 of [1].

    TODO

    References
    ----------
    [1] Christian Dittmar, Patricio Lopez-Serrano, Meinard Müller
    Unifying Local and Global Methods for Harmonic-Percussive Source Separation
    In Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    W: array-like
        NMF templates given in matrix/tensor form

    H: array-like
        NMF activations given as matrix

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
    percWeight = percussiveness_estimation(W).reshape(1, -1)

    # promote harmonic sustained gains
    Hh = median_filter(H, footprint=parameter['Kern'], mode='constant')

    # promote decaying impulses gains
    Hp = nema(H, decay)

    # make weighted sum according to percussiveness measure
    H_out = Hh * (1 - percWeight.T) + Hp * percWeight.T

    return W, H_out

def init_templates(num_comp: int,
                   num_bins: int,
                   pitches: Union[List[int], None],
                   strategy='random',
                   pitch_tol_up: float = 0.75,
                   pitch_tol_down: float = 0.75,
                   num_harmonics: int = 25,
                   num_template_frames: int = 1,
                   delta_F = None) -> np.ndarray:
    """Implements different initialization strategies for NMF templates. The strategies 'random' and 'uniform' are
    self-explaining. The strategy 'pitched' uses comb-filter templates as described in [1]. The strategy 'drums' uses
     pre-extracted, averaged spectra of desired drum types [2].

    References
    ----------
    [1] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert and Meinard Mueller
    "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM) Barcelona, Spain, 2013.

    [2] Christian Dittmar and Meinard Müller
    "Reverse Engineering the Amen Break - Score-informed Separation and Restoration applied to Drum Recordings"
    IEEE/ACM Transactions on Audio, Speech, and Language Processing, 24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numBins           Number of frequency bins
        numTemplateFrames Number of time frames for 2D-templates
        pitches           Optional array of MIDI pitch values
        drumTypes         Optional list of drum type strings

    strategy: str
        String describing the initialization strategy

    Returns
    -------
    initW: array-like
        List with the desired templates
    """
    # check parameters
    init_W = list()

    if strategy == 'random':
        # fix random seed
        np.random.seed(42)

        for k in range(num_comp):
            init_W.append(np.random.rand(num_bins, num_template_frames))

    elif strategy == 'uniform':
        for k in range(num_comp):
            init_W.append(np.ones((num_bins, num_template_frames)))

    elif strategy == 'pitched':
        unique_pitches = np.unique(pitches)

        # needs to be overwritten
        num_comp = unique_pitches.size

        for k in range(unique_pitches.size):
            # initialize as zeros
            init_W.append(EPS + np.zeros((num_bins, num_template_frames)))

            # then insert non-zero entries in bands around hypothetic harmonics
            cur_pitch_freq_lower_hz = midi2freq(unique_pitches[k] - pitch_tol_down)
            curPitchFreqUpper_Hz = midi2freq(unique_pitches[k] + pitch_tol_up)

            for g in range(num_harmonics):
                curr_pitch_freq_lower_bins = (g + 1) * cur_pitch_freq_lower_hz / delta_F
                curr_pitch_freq_upper_bins = (g + 1) * curPitchFreqUpper_Hz / delta_F

                bin_range = np.arange(int(round(curr_pitch_freq_lower_bins)) - 1, int(round(curr_pitch_freq_upper_bins)))
                bin_range = bin_range[0:num_bins]

                # insert 1/f intensity
                init_W[k][bin_range, :] = 1/(g+1)

    elif strategy == 'drums':
        dict_W = load_matlab_dict('../data/dictW.mat', 'dictW')

        if num_bins == dict_W.shape[0]:
            for k in range(dict_W.shape[1]):
                init_W.append(dict_W[:, k].reshape(-1, 1) * np.linspace(1, 0.1, num_template_frames))

        # needs to be overwritten
        num_comp = len(init_W)

    else:
        raise ValueError('Invalid strategy.')

    # do final normalization
    for k in range(num_comp):
        init_W[k] /= (EPS + init_W[k].sum())

    return init_W


def init_activations(num_comp: int,
                     num_frames: int,
                     strategy: str,
                     delta_T: float,
                     pitches: Union[List[int], None],
                     decay: float = 0.75,
                     onset_offset_tol: float = 0.025,
                     onsets: List = None,
                     durations: List = None,
                     drums: List = None):
    """Implements different initialization strategies for NMF activations. The
    strategies 'random' and 'uniform' are self-explaining. The strategy
    'pitched' places gate-like activations at the frames, where certain notes
    are active in the ground truth transcription [2]. The strategy
    'drums' places decaying impulses at the frames where drum onsets are given
    in the ground truth transcription [3].

    References
    ----------
    [2] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
    and Meinard Müller "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
    Barcelona, Spain, 2013.

    [3] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numFrames         Number of time frames
        deltaT            The temporal resolution
        pitches           Optional array of MIDI pitch values
        onsets            Optional array of note onsets (in seconds)
        durations         Optional array of note durations (in seconds)
        drums             Optional array of drum type indices
        decay             Optional array of decay values per component
        onsetOffsetTol    Optional parameter giving the onset / offset

    strategy: str
        String describing the intialization strategy

    Returns
    -------
    initH: array-like
        Array with initial activation functions
    """

    if strategy == 'random':
        np.random.seed(42)
        init_H = np.random.rand(num_comp, num_frames)

    elif strategy == 'uniform':
        init_H = np.ones((num_comp, num_frames))

    elif strategy == 'pitched':
        uniquePitches = np.unique(pitches)

        # overwrite
        num_comp = uniquePitches.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        for k in range(uniquePitches.size):

            # find corresponding note onsets and durations
            ind = np.nonzero(pitches == uniquePitches[k])[0]

            # insert activations
            for g in range(len(ind)):
                curr_ind = ind[g]

                note_start_in_seconds = onsets[curr_ind]
                note_end_in_seconds = note_start_in_seconds + durations[curr_ind]

                note_start_in_seconds -= onset_offset_tol
                note_end_in_seconds += onset_offset_tol

                note_start_in_frames = int(round(note_start_in_seconds / delta_T))
                note_ende_in_frames = int(round(note_end_in_seconds / delta_T))

                frameRange = np.arange(note_start_in_frames, note_ende_in_frames + 1)
                frameRange = frameRange[frameRange >= 0]
                frameRange = frameRange[frameRange <= num_frames]

                # insert gate-like activation
                init_H[k, frameRange-1] = 1

    elif strategy == 'drums':
        uniqueDrums = np.unique(drums)

        # overwrite
        num_comp = uniqueDrums.size

        # initialize activations with very small values
        init_H = EPS + np.zeros((num_comp, num_frames))

        # sanity check
        if uniqueDrums.size == num_comp:

            # insert impulses at onset positions
            for k in range(len(uniqueDrums)):
                currOns = np.nonzero(drums == uniqueDrums[k])[0]
                currOns = onsets[currOns]
                currOns = np.round(currOns/delta_T).astype(np.int)
                currOns = currOns[currOns >= 0]
                currOns = currOns[currOns <= num_frames]

                init_H[uniqueDrums[k].astype(int)-1, currOns-1] = 1

            # add exponential decay
            init_H = nema(init_H, decay)

    else:
        raise ValueError('Invalid strategy.')

    return init_H


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

    col_vec:

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
                             delta_T: float,
                             delta_F: float,
                             start_sec: float = None,
                             end_sec: float = None,
                             font_size: float = 11):
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components, this function provides a
    visualization.

    Parameters
    ----------
    comp_A: list
        List with R individual component magnitude spectrograms.
    delta_T: float
        Temporal resolution
    delta_F: float
        Spectral resolution
    start_sec: float
        Where to zoom in on the time axis
    end_sec: float
        Where to zoom in on the time axis
    font_size: float
        Font size of the figure.

    Returns
    -------
    fh: The figure handle
    """
    # get spectrogram dimensions
    num_lin_bins, num_frames = comp_A[0].shape

    start_sec = delta_T if start_sec is None else start_sec
    end_sec = num_frames * delta_T if end_sec is None else end_sec

    # plot MMF / NMFD components
    # map template spectrograms to a logarithmically - spaced frequency
    # and logarithmic magnitude compression

    log_freq_log_mag_comp_A, log_freq_axis = log_freq_log_mag(A=comp_A, delta_F=delta_F)
    num_log_bins = len(log_freq_axis)

    time_axis = np.arange(num_frames) * delta_T
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
                             delta_T: float = None,
                             delta_F: np.ndarray = None,
                             start_sec: float = None,
                             end_sec: float = None,
                             font_size: float = 11):
    """Given a non-negative matrix V, and its non non-negative NMF or NMFD components, this function provides a
    visualization.

    Parameters
    ----------
    V: np.ndarray
        K x M non-negative target matrix, in our case, this is usually sa magnitude spectrogram
    W: np.ndarray
        K X R matrix of learned template matrices
    H: np.ndarray
        R X M matrix of learned activations
    comp_V: np.ndarray
        Matrix with R individual component magnitude spectrograms
    log_comp: float
        Factor to control the logarithmic magnitude compression
    delta_T: float
        Temporal resolution
    delta_F: float
        Spectral resolution
    start_sec: float
        Where to zoom in on the time axis
    end_sec: float
        Where to zoom in on the time axis
    font_size: float
        Font size of the figure

    Returns
    -------
    fh: The figure handle
    """
    R = H.shape[0]
    num_lin_bins, num_frames = V.shape
    comp_col_vec = __set_comp_vol_vec(R)

    # plot MMF / NMFD components
    # map the target and the templates to a logarithmically-spaced frequency
    # and logarithmic magnitude compression
    log_freq_log_magV, log_freq_axis = log_freq_log_mag(V, delta_F=delta_F, log_comp=log_comp)
    num_log_bins = len(log_freq_axis)

    log_freq_log_mag_W, log_freq_axis = log_freq_log_mag(W, delta_F=delta_F, log_comp=log_comp)

    if comp_V is not None:
        log_freq_log_mag_comp_V, log_freq_axis = log_freq_log_mag(A=comp_V, delta_F=delta_F, log_comp=log_comp)
    else:
        log_freq_log_mag_comp_V = [np.array(log_freq_log_magV)]  # simulate one component

    time_axis = np.arange(num_frames) * delta_T
    freq_axis = np.arange(num_log_bins)

    # subsample freq axis
    sub_samp = np.where(np.mod(log_freq_axis.astype(np.float32), 55) < 0.001)[0]
    sub_samp_freq_axis = log_freq_axis[np.mod(log_freq_axis.astype(np.float32), 55) < 0.001]

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
            xcoord = 0.5 / delta_F + np.concatenate([time_axis.reshape(1, -1), np.fliplr(time_axis.reshape(1, -1))], axis=1)
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
        return np.array([[1, 0, 0], [0, 0.5, 0.5]], dtype=np.float)
    elif R == 3:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    elif R == 4:
        return np.array([[1, 0, 1], [1, 0.5, 0], [0, 1, 0], [0, 0.5, 1]], dtype=np.float)
    else:
        return np.tile(np.array([0.5, 0.5, 0.5]), (R, 1)).astype(np.float)
